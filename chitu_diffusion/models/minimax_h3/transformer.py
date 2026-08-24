from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from chitu_diffusion.parallel.linear import ColumnParallelLinear
from chitu_diffusion.parallel.tensor_parallel import (
    get_tp_group,
    get_tp_world_size,
)
from chitu_diffusion.parallel.topology import UspTopology

from .attention import MiniMaxH3Attention
from .config import MiniMaxH3DiTConfig
from .layers import (
    MiniMaxH3AdalnProj,
    MiniMaxH3MLP,
    MiniMaxH3TimeEmbedder,
    modulate_gate,
    modulate_scale_shift,
)
from .rope import MiniMaxH3Rope


@dataclass(frozen=True, slots=True)
class MiniMaxH3PackedForwardInputs:
    """Runtime-neutral packed rows consumed by the optimized H3 DiT."""

    video_rows: torch.Tensor
    audio_rows: torch.Tensor
    refined_prompt_embeds: torch.Tensor
    img_pos: torch.Tensor
    audio_pos: torch.Tensor
    text_pos: torch.Tensor
    global_sequence_length: int
    unique_timesteps: torch.Tensor
    inverse_indices: torch.Tensor
    token_tags: torch.Tensor
    position_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int
    row_start: int = 0
    row_stop: int | None = None


class MiniMaxH3TokenRefinerBlock(nn.Module):
    def __init__(
        self,
        config: MiniMaxH3DiTConfig,
        *,
        attention_backend: str,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(
            config.hidden_size,
            eps=config.norm_eps,
            dtype=torch.bfloat16,
            device=device,
        )
        self.norm2 = nn.RMSNorm(
            config.hidden_size,
            eps=config.norm_eps,
            dtype=torch.bfloat16,
            device=device,
        )
        self.attn = MiniMaxH3Attention(
            config, attention_backend=attention_backend, device=device
        )
        self.mlp = MiniMaxH3MLP(config, device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            rope_frequencies=None,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


class MiniMaxH3TokenRefiner(nn.Module):
    def __init__(
        self,
        config: MiniMaxH3DiTConfig,
        *,
        attention_backend: str,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                MiniMaxH3TokenRefinerBlock(
                    config,
                    attention_backend=attention_backend,
                    device=device,
                )
                for _ in range(config.token_refiner_num_layers)
            ]
        )
        self.final_norm = nn.RMSNorm(
            config.hidden_size,
            eps=config.final_norm_eps,
            dtype=torch.bfloat16,
            device=device,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        return self.final_norm(hidden_states)


class MiniMaxH3DiTBlock(nn.Module):
    def __init__(
        self,
        config: MiniMaxH3DiTConfig,
        *,
        attention_backend: str,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(
            config.hidden_size,
            eps=config.norm_eps,
            dtype=torch.bfloat16,
            device=device,
        )
        self.norm2 = nn.RMSNorm(
            config.hidden_size,
            eps=config.norm_eps,
            dtype=torch.bfloat16,
            device=device,
        )
        self.attn = MiniMaxH3Attention(
            config, attention_backend=attention_backend, device=device
        )
        self.mlp = MiniMaxH3MLP(config, device=device)
        self.adaln_proj = MiniMaxH3AdalnProj(
            config,
            config.adaln_out_features,
            expand_ratio=6,
            modality_count=3,
            device=device,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        adaln_input: torch.Tensor,
        combined_indices: torch.Tensor,
        rope_frequencies: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        ulysses_topology: UspTopology | None,
    ) -> torch.Tensor:
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaln_proj(adaln_input)
        residual = hidden_states
        hidden = modulate_scale_shift(
            self.norm1(hidden_states),
            shift_msa,
            scale_msa,
            combined_indices,
        )
        hidden = self.attn(
            hidden,
            rope_frequencies=rope_frequencies,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            ulysses_topology=ulysses_topology,
        )
        hidden_states = modulate_gate(
            residual, gate_msa, hidden, combined_indices
        )
        residual = hidden_states
        hidden = modulate_scale_shift(
            self.norm2(hidden_states),
            shift_mlp,
            scale_mlp,
            combined_indices,
        )
        return modulate_gate(
            residual,
            gate_mlp,
            self.mlp(hidden),
            combined_indices,
        )


class MiniMaxH3FinalLayer(nn.Module):
    def __init__(
        self,
        config: MiniMaxH3DiTConfig,
        *,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(
            config.hidden_size,
            eps=config.final_norm_eps,
            dtype=torch.bfloat16,
            device=device,
        )
        self.adaln_proj = MiniMaxH3AdalnProj(
            config,
            config.final_adaln_out_features,
            expand_ratio=2,
            modality_count=1,
            device=device,
        )
        self.video_out = ColumnParallelLinear(
            config.hidden_size,
            config.video_patch_dim,
            params_dtype=torch.float32,
            device=device,
        )
        self.audio_out = ColumnParallelLinear(
            config.hidden_size,
            config.audio_latents_dim,
            params_dtype=torch.float32,
            device=device,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        adaln_input: torch.Tensor,
        inverse_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift, scale = self.adaln_proj(adaln_input)
        hidden = modulate_scale_shift(
            self.norm(hidden_states), shift, scale, inverse_indices
        ).to(torch.float32)
        video, _ = self.video_out(hidden)
        audio, _ = self.audio_out(hidden)
        return video, audio


class MiniMaxH3DiTModel(nn.Module):
    def __init__(
        self,
        config: MiniMaxH3DiTConfig,
        *,
        attention_backend: str = "auto",
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        config.validate_parallel(get_tp_world_size())
        self.video_patch_proj = ColumnParallelLinear(
            config.video_patch_dim,
            config.hidden_size,
            gather_output=True,
            params_dtype=torch.float32,
            device=device,
        )
        self.audio_patch_proj = ColumnParallelLinear(
            config.audio_latents_dim,
            config.hidden_size,
            gather_output=True,
            params_dtype=torch.float32,
            device=device,
        )
        self.condition_proj = ColumnParallelLinear(
            config.text_dim,
            config.hidden_size,
            gather_output=True,
            params_dtype=torch.bfloat16,
            device=device,
        )
        self.time_embedder = MiniMaxH3TimeEmbedder(config, device=device)
        self.rope = MiniMaxH3Rope(config.rope_inv_freq_len, device=device)
        self.token_refiner = MiniMaxH3TokenRefiner(
            config,
            attention_backend=attention_backend,
            device=device,
        )
        self.blocks = nn.ModuleList(
            [
                MiniMaxH3DiTBlock(
                    config,
                    attention_backend=attention_backend,
                    device=device,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_layer = MiniMaxH3FinalLayer(config, device=device)

    def refine_prompt_embeds(
        self,
        prompt_embeds: torch.Tensor,
        *,
        live_length: int | None = None,
    ) -> torch.Tensor:
        live = int(prompt_embeds.shape[0] if live_length is None else live_length)
        prompt = prompt_embeds[:live].to(torch.bfloat16)
        hidden, _ = self.condition_proj(prompt)
        cu = torch.tensor(
            [0, live, live], device=prompt.device, dtype=torch.int32
        )
        return self.token_refiner(
            hidden, cu_seqlens=cu, max_seqlen=max(live, 1)
        )

    def embed_packed(
        self,
        *,
        video_rows: torch.Tensor,
        audio_rows: torch.Tensor,
        refined_prompt_embeds: torch.Tensor,
        img_pos: torch.Tensor,
        audio_pos: torch.Tensor,
        text_pos: torch.Tensor,
        global_sequence_length: int,
        row_start: int = 0,
        row_stop: int | None = None,
    ) -> torch.Tensor:
        row_stop = global_sequence_length if row_stop is None else row_stop
        output = torch.zeros(
            row_stop - row_start,
            self.config.hidden_size,
            device=video_rows.device,
            dtype=torch.bfloat16,
        )

        def scatter(
            positions: torch.Tensor,
            values: torch.Tensor,
        ) -> None:
            selected = (positions >= row_start) & (positions < row_stop)
            source_ids = selected.nonzero(as_tuple=False).flatten()
            if source_ids.numel():
                local_ids = positions.index_select(0, source_ids) - row_start
                output.index_copy_(0, local_ids, values.index_select(0, source_ids))

        video_embed, _ = self.video_patch_proj(video_rows.to(torch.float32))
        audio_embed, _ = self.audio_patch_proj(audio_rows.to(torch.float32))
        scatter(img_pos, video_embed.to(torch.bfloat16))
        scatter(audio_pos, audio_embed.to(torch.bfloat16))
        scatter(text_pos, refined_prompt_embeds.to(torch.bfloat16))
        return output

    def forward_hidden(
        self,
        hidden_states: torch.Tensor,
        *,
        unique_timesteps: torch.Tensor,
        inverse_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        ulysses_topology: UspTopology | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        degree = 1 if ulysses_topology is None else ulysses_topology.ulysses_degree
        self.config.validate_parallel(get_tp_world_size(), degree)
        if ulysses_topology is not None and ulysses_topology.ring_degree != 1:
            raise NotImplementedError("MiniMax-H3 supports pure Ulysses only")
        local_tokens = hidden_states.shape[0]
        if position_ids.shape[-2] != local_tokens:
            if position_ids.shape[-2] != local_tokens * degree:
                raise ValueError("position_ids must be local or global sequence length")
            rank = 0 if ulysses_topology is None else ulysses_topology.ulysses_rank
            position_ids = position_ids[
                ..., rank * local_tokens : (rank + 1) * local_tokens, :
            ]
        if inverse_indices.numel() != local_tokens:
            rank = 0 if ulysses_topology is None else ulysses_topology.ulysses_rank
            inverse_indices = inverse_indices[
                rank * local_tokens : (rank + 1) * local_tokens
            ]
            token_tags = token_tags[
                rank * local_tokens : (rank + 1) * local_tokens
            ]
        rope_frequencies = self.rope(position_ids).to(hidden_states.device)
        time_embedding = self.time_embedder(unique_timesteps)
        adaln_input = F.silu(time_embedding).to(torch.bfloat16)
        combined_indices = (
            inverse_indices.to(torch.long) * 3
            + token_tags.to(torch.long).clamp(min=0)
        )
        hidden = hidden_states
        for block in self.blocks:
            hidden = block(
                hidden,
                adaln_input=adaln_input,
                combined_indices=combined_indices,
                rope_frequencies=rope_frequencies,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                ulysses_topology=ulysses_topology,
            )
        return self.final_layer(
            hidden,
            adaln_input=adaln_input,
            inverse_indices=inverse_indices.to(torch.long),
        )

    def forward_packed(
        self,
        inputs: MiniMaxH3PackedForwardInputs,
        *,
        ulysses_topology: UspTopology | None = None,
        gather_tp_output: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed packed media/text rows and execute the shared DiT core.

        Sequence-parallel gathering is deliberately left to the host runtime;
        TP head outputs can be gathered here because both Chitu EPE and SGLang
        require full 96/32-wide prediction rows at the scheduler boundary.
        """

        hidden_states = self.embed_packed(
            video_rows=inputs.video_rows,
            audio_rows=inputs.audio_rows,
            refined_prompt_embeds=inputs.refined_prompt_embeds,
            img_pos=inputs.img_pos,
            audio_pos=inputs.audio_pos,
            text_pos=inputs.text_pos,
            global_sequence_length=inputs.global_sequence_length,
            row_start=inputs.row_start,
            row_stop=inputs.row_stop,
        )
        video, audio = self.forward_hidden(
            hidden_states,
            unique_timesteps=inputs.unique_timesteps,
            inverse_indices=inputs.inverse_indices,
            token_tags=inputs.token_tags,
            position_ids=inputs.position_ids,
            cu_seqlens=inputs.cu_seqlens,
            max_seqlen=inputs.max_seqlen,
            ulysses_topology=ulysses_topology,
        )
        if gather_tp_output:
            video = self.gather_tp_output(video)
            audio = self.gather_tp_output(audio)
        return video, audio

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        unique_timesteps: torch.Tensor,
        inverse_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        ulysses_topology: UspTopology | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_hidden(
            hidden_states,
            unique_timesteps=unique_timesteps,
            inverse_indices=inverse_indices,
            token_tags=token_tags,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            ulysses_topology=ulysses_topology,
        )

    @staticmethod
    def gather_tp_output(output: torch.Tensor) -> torch.Tensor:
        degree = get_tp_world_size()
        if degree == 1:
            return output
        pieces = [torch.empty_like(output) for _ in range(degree)]
        dist.all_gather(pieces, output.contiguous(), group=get_tp_group())
        return torch.cat(pieces, dim=-1).contiguous()

    def post_load_weights(self) -> None:
        for name in FP32_PARAMETER_NAMES:
            parameter = self.get_parameter(name)
            if parameter.dtype != torch.float32:
                raise ValueError(f"{name} must remain fp32")
        if self.rope.inv_freq.dtype != torch.float32:
            raise ValueError("rope.inv_freq must remain fp32")


FP32_PARAMETER_NAMES = (
    "video_patch_proj.weight",
    "video_patch_proj.bias",
    "audio_patch_proj.weight",
    "audio_patch_proj.bias",
    "time_embedder.proj_in.weight",
    "time_embedder.proj_in.bias",
    "time_embedder.proj_out.weight",
    "time_embedder.proj_out.bias",
    "final_layer.video_out.weight",
    "final_layer.video_out.bias",
    "final_layer.audio_out.weight",
    "final_layer.audio_out.bias",
)

