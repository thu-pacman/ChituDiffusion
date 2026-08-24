from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from chitu_diffusion.models.minimax_h3.config import MiniMaxH3DiTConfig
from chitu_diffusion.models.minimax_h3.loader import reorder_grouped_qkv_to_qkv
from chitu_diffusion.models.minimax_h3.transformer import (
    MiniMaxH3DiTModel as _ChituMiniMaxH3DiTModel,
    MiniMaxH3PackedForwardInputs,
)
from chitu_diffusion.parallel.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from chitu_diffusion.parallel.tensor_parallel import (
    adopt_external_tensor_parallel_group,
)
from chitu_diffusion.parallel.topology import UspTopology


_FORWARD_SUPPORTED_KWARGS = frozenset(
    {
        "x",
        "audio_x",
        "img_position_ids",
        "rope_cache",
        "unique_timesteps",
        "inverse_indices",
        "update_mask",
        "update_audio_mask",
        "token_tags",
        "block_token_tags",
        "block_combined_indices",
        "skip_mask_out_condition",
        "prompt_embeds",
        "refined_prompt_embeds_length",
        "img_pos_info",
        "audio_pos_info",
        "text_pos_info",
        "img_pos_for_infer_output_info",
        "local_embedding_layout",
        "packed_seq_params",
        "refiner_packed_seq_params",
    }
)
_BCG_ONLY_KWARGS = (
    "block_token_tags",
    "block_combined_indices",
    "local_embedding_layout",
)
_CONFIG_FIELDS = tuple(MiniMaxH3DiTConfig.__dataclass_fields__)
_LINEAR_TYPES = (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)


def _required(kwargs: Mapping[str, Any], key: str) -> Any:
    value = kwargs.get(key)
    if value is None:
        raise ValueError(f"MiniMaxH3DiTModel.forward requires kwarg {key!r}")
    return value


def _field(value: Any, name: str, owner: str) -> Any:
    result = value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)
    if result is None:
        raise ValueError(f"{owner}.{name} is required")
    return result


def _position_ids(value: Any, owner: str) -> torch.Tensor:
    return _field(value, "position_ids", owner).reshape(-1).to(torch.long)


def _sglang_parallel_state() -> Any:
    from sglang.multimodal_gen.runtime.distributed import parallel_state

    return parallel_state


def _adopt_sglang_tp_group() -> None:
    state = _sglang_parallel_state()
    if not state.model_parallel_is_initialized():
        rank = dist.get_rank() if dist.is_initialized() else 0
        adopt_external_tensor_parallel_group(
            ranks=(rank,),
            rank=rank,
            process_group=None,
        )
        return
    group = state.get_tp_group()
    adopt_external_tensor_parallel_group(
        ranks=tuple(group.ranks),
        rank=int(group.rank),
        process_group=group.device_group,
    )


def _sglang_ulysses_topology() -> UspTopology | None:
    state = _sglang_parallel_state()
    if not state.model_parallel_is_initialized():
        return None
    ulysses_degree = int(state.get_ulysses_parallel_world_size())
    ring_degree = int(state.get_ring_parallel_world_size())
    if ring_degree != 1:
        raise NotImplementedError(
            "Chitu MiniMax-H3 adapter supports pure Ulysses only; "
            "set --ring-degree 1"
        )
    if ulysses_degree == 1:
        return None

    sp_group = state.get_sp_group()
    ulysses_group = sp_group.ulysses_group
    ring_group = sp_group.ring_group
    if ulysses_group is None:
        raise RuntimeError("SGLang Ulysses process group is not initialized")
    ulysses_ranks = tuple(dist.get_process_group_ranks(ulysses_group))
    ring_ranks = (
        tuple(dist.get_process_group_ranks(ring_group))
        if ring_group is not None
        else (int(sp_group.rank),)
    )
    return UspTopology(
        lane_ranks=tuple(sp_group.ranks),
        ulysses_ranks=ulysses_ranks,
        ring_ranks=ring_ranks,
        ulysses_rank=int(state.get_ulysses_parallel_rank()),
        ring_rank=int(state.get_ring_parallel_rank()),
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        ulysses_process_group=ulysses_group,
        ring_process_group=ring_group,
    )


def _to_chitu_config(config: Any, hf_config: Mapping[str, Any]) -> MiniMaxH3DiTConfig:
    arch = getattr(config, "arch_config", config)
    values: dict[str, Any] = {}
    for name in _CONFIG_FIELDS:
        value = getattr(arch, name, hf_config.get(name))
        if value is not None:
            values[name] = tuple(value) if name == "patch_size" else value
    return MiniMaxH3DiTConfig(**values)


def _rope_cos_sin_cache(frequencies: torch.Tensor) -> torch.Tensor:
    half = frequencies.shape[-1] // 2
    return (
        torch.cat(
            (
                torch.cos(frequencies[:, :half]),
                torch.sin(frequencies[:, :half]),
            ),
            dim=-1,
        )
        .to(torch.bfloat16)
        .contiguous()
    )


class MiniMaxH3DiTModel(_ChituMiniMaxH3DiTModel):
    """SGLang constructor and packed-forward contract over Chitu's H3 DiT."""

    param_names_mapping: dict[str, str] = {}
    reverse_param_names_mapping: dict[str, str] = {}
    lora_param_names_mapping: dict[str, str] = {}
    _compile_conditions: list[Any] = []
    _fsdp_mixed_dtype_params = True

    def __init__(
        self,
        config: Any,
        hf_config: dict[str, Any],
        quant_config: Any = None,
    ) -> None:
        if quant_config is not None:
            raise NotImplementedError(
                "Chitu MiniMax-H3 registry adapter does not support SGLang "
                "quantized transformer loading"
            )
        _adopt_sglang_tp_group()
        chitu_config = _to_chitu_config(config, hf_config)
        super().__init__(chitu_config, attention_backend="auto")
        self.sglang_config = config
        self.hf_config = dict(hf_config)
        self.arch = getattr(config, "arch_config", config)
        self.hidden_size = chitu_config.hidden_size
        self.num_attention_heads = chitu_config.num_attention_heads
        self.num_channels_latents = chitu_config.latents_dim
        self.layer_names = ["blocks"]
        self._install_sglang_weight_loaders()

    @property
    def _fsdp_shard_conditions(self) -> list[Any]:
        raise NotImplementedError(
            "Chitu MiniMax-H3 registry adapter does not support SGLang FSDP; "
            "launch with --no-use-fsdp-inference"
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _install_sglang_weight_loaders(self) -> None:
        for name, module in self.named_modules():
            if not isinstance(module, _LINEAR_TYPES):
                continue
            for parameter_name in ("weight", "bias"):
                parameter = getattr(module, parameter_name, None)
                if isinstance(parameter, nn.Parameter):
                    parameter.weight_loader = module.weight_loader

            if not (
                name.endswith(".attn.qkv_proj")
                and isinstance(module, MergedColumnParallelLinear)
            ):
                continue
            base_loader = module.weight_loader
            num_heads = self.config.num_attention_heads
            head_dim = self.config.attention_head_dim

            @torch.no_grad()
            def load_grouped_qkv(
                parameter: nn.Parameter,
                loaded_weight: torch.Tensor,
                *,
                _base_loader=base_loader,
                _num_heads=num_heads,
                _head_dim=head_dim,
            ) -> None:
                reordered = reorder_grouped_qkv_to_qkv(
                    loaded_weight,
                    num_heads=_num_heads,
                    head_dim=_head_dim,
                )
                _base_loader(parameter, reordered)

            module.weight.weight_loader = load_grouped_qkv

        for parameter in self.parameters():
            parameter.missing_param_init = "error"

    def refine_prompt_embeds(
        self,
        prompt_embeds: torch.Tensor,
        refiner_cu_seqlens: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if refiner_cu_seqlens.numel() < 2:
            raise ValueError("refiner_cu_seqlens must contain at least two entries")
        live_length = int(refiner_cu_seqlens[1].item())
        if live_length <= 0 or live_length > int(prompt_embeds.shape[0]):
            raise ValueError(
                "refiner live text length must be within prompt_embeds rows"
            )
        return super().refine_prompt_embeds(
            prompt_embeds.to(device),
            live_length=live_length,
        )

    def build_rope_cache(
        self,
        img_position_ids: torch.Tensor,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if img_position_ids.ndim != 3 or img_position_ids.shape[0] != 1:
            raise ValueError(
                "img_position_ids must be [1, S, 3], got "
                f"{list(img_position_ids.shape)}"
            )
        topology = _sglang_ulysses_topology()
        degree = 1 if topology is None else topology.ulysses_degree
        rank = 0 if topology is None else topology.ulysses_rank
        sequence_length = int(img_position_ids.shape[1])
        if sequence_length % degree:
            raise ValueError(
                f"packed seq_len {sequence_length} is not divisible by "
                f"Ulysses world size {degree}"
            )
        local_length = sequence_length // degree
        start = rank * local_length
        frequencies = self.rope(
            img_position_ids[:, start : start + local_length]
        ).to(device)
        return (
            _rope_cos_sin_cache(frequencies),
            torch.arange(local_length, device=device, dtype=torch.long),
        )

    def forward(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        unexpected = sorted(set(kwargs) - _FORWARD_SUPPORTED_KWARGS)
        if unexpected:
            raise TypeError(
                "MiniMaxH3DiTModel.forward received unexpected kwargs: "
                f"{unexpected}; supported kwargs: "
                f"{sorted(_FORWARD_SUPPORTED_KWARGS)}"
            )
        bcg = [key for key in _BCG_ONLY_KWARGS if kwargs.get(key) is not None]
        if bcg or bool(kwargs.get("skip_mask_out_condition", False)):
            raise NotImplementedError(
                "Chitu MiniMax-H3 registry adapter does not support SGLang "
                f"breakable CUDA graph inputs: {bcg or ['skip_mask_out_condition']}"
            )

        x = _required(kwargs, "x")
        audio_x = _required(kwargs, "audio_x")
        position_ids = _required(kwargs, "img_position_ids")
        unique_timesteps = _required(kwargs, "unique_timesteps").reshape(-1)
        inverse_indices = _required(kwargs, "inverse_indices").reshape(-1)
        token_tags = _required(kwargs, "token_tags").reshape(-1)
        prompt_embeds = _required(kwargs, "prompt_embeds")
        update_mask = _required(kwargs, "update_mask").reshape(-1)
        img_pos = _position_ids(_required(kwargs, "img_pos_info"), "img_pos_info")
        audio_pos = _position_ids(
            _required(kwargs, "audio_pos_info"), "audio_pos_info"
        )
        text_pos = _position_ids(
            _required(kwargs, "text_pos_info"), "text_pos_info"
        )
        infer_pos = _position_ids(
            _required(kwargs, "img_pos_for_infer_output_info"),
            "img_pos_for_infer_output_info",
        )
        packed = _required(kwargs, "packed_seq_params")
        cu_seqlens = _field(packed, "cu_seqlens_q", "packed_seq_params").to(
            torch.int32
        )
        max_seqlen = int(_field(packed, "max_seqlen_q", "packed_seq_params"))

        if x.ndim != 3 or x.shape[0] != 1:
            raise ValueError(f"x must be [1, S, C], got {list(x.shape)}")
        if audio_x.ndim != 3 or audio_x.shape[0] != 1:
            raise ValueError(
                f"audio_x must be [1, S, C], got {list(audio_x.shape)}"
            )
        sequence_length = int(x.shape[1])
        if inverse_indices.numel() != sequence_length:
            raise ValueError("inverse_indices must cover the packed sequence")
        if token_tags.numel() != sequence_length:
            raise ValueError("token_tags must cover the packed sequence")

        live_length = kwargs.get("refined_prompt_embeds_length")
        if live_length is None:
            refiner = _required(kwargs, "refiner_packed_seq_params")
            refiner_cu = _field(
                refiner,
                "cu_seqlens_q",
                "refiner_packed_seq_params",
            )
            refined = self.refine_prompt_embeds(
                prompt_embeds,
                refiner_cu,
                device=x.device,
            )
        else:
            live_length = int(
                live_length.item() if torch.is_tensor(live_length) else live_length
            )
            if live_length <= 0 or live_length > int(prompt_embeds.shape[0]):
                raise ValueError("refined_prompt_embeds_length is out of range")
            refined = prompt_embeds[:live_length].to(
                device=x.device,
                dtype=torch.bfloat16,
            )

        video_rows = x.reshape(-1, x.shape[-1]).index_select(
            0, img_pos.to(x.device)
        )
        audio_rows = audio_x.reshape(-1, audio_x.shape[-1]).index_select(
            0, audio_pos.to(audio_x.device)
        )
        topology = _sglang_ulysses_topology()
        degree = 1 if topology is None else topology.ulysses_degree
        rank = 0 if topology is None else topology.ulysses_rank
        if sequence_length % degree:
            raise ValueError("packed sequence must divide the Ulysses degree")
        local_length = sequence_length // degree
        row_start = rank * local_length
        row_stop = row_start + local_length
        video, audio = self.forward_packed(
            MiniMaxH3PackedForwardInputs(
                video_rows=video_rows,
                audio_rows=audio_rows,
                refined_prompt_embeds=refined,
                img_pos=img_pos.to(x.device),
                audio_pos=audio_pos.to(x.device),
                text_pos=text_pos[: refined.shape[0]].to(x.device),
                global_sequence_length=sequence_length,
                row_start=row_start,
                row_stop=row_stop,
                unique_timesteps=unique_timesteps.to(x.device),
                inverse_indices=inverse_indices.to(x.device),
                token_tags=token_tags.to(x.device),
                position_ids=position_ids.to(x.device),
                cu_seqlens=cu_seqlens.to(x.device),
                max_seqlen=max_seqlen,
            ),
            ulysses_topology=topology,
            gather_tp_output=False,
        )

        if topology is not None:
            pieces = [
                torch.empty_like(torch.cat((video, audio), dim=-1))
                for _ in range(topology.ulysses_degree)
            ]
            dist.all_gather(
                pieces,
                torch.cat((video, audio), dim=-1).contiguous(),
                group=topology.ulysses_process_group,
            )
            video_width = video.shape[-1]
            gathered = torch.cat(pieces, dim=0)
            video, audio = gathered.split(
                (video_width, gathered.shape[-1] - video_width),
                dim=-1,
            )

        video = self.gather_tp_output(
            video.index_select(0, infer_pos.to(video.device))
        )
        audio = self.gather_tp_output(
            audio.index_select(0, audio_pos.to(audio.device))
        )
        video = video * update_mask.to(video.device).unsqueeze(-1)
        update_audio_mask = kwargs.get("update_audio_mask")
        if update_audio_mask is not None:
            audio = audio * update_audio_mask.reshape(-1).to(audio.device).unsqueeze(-1)
        return video, audio


EntryClass = MiniMaxH3DiTModel

__all__ = [
    "EntryClass",
    "MiniMaxH3DiTModel",
    "_FORWARD_SUPPORTED_KWARGS",
]

