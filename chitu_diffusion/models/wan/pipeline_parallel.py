"""Wan 2.1 fine-grained pipeline execution on the current Diffusers graph."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from accelerate import init_empty_weights
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_wan import WanAttnProcessor
from torch import nn

from ...parallel.cp.nccl.agkv import all_gather_sequence_pair
from ...parallel.pp import (
    FppAttentionContext,
    FppState,
    PipelineTopology,
    PipelineTransport,
    current_fpp_attention_context,
    fpp_attention_context,
    partition_bounds,
)
from ...parallel.tp.loader import load_tensor_parallel_checkpoint
from .attention import _apply_rotary


class WanFppAttnProcessor(WanAttnProcessor):
    def __init__(self, layer: int) -> None:
        self.layer = layer

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if encoder_hidden_states is not None or attention_mask is not None:
            raise NotImplementedError(
                "Wan FPP implements unmasked video self-attention"
            )
        context = current_fpp_attention_context()
        if getattr(attn, "fused_projections", False):
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
        query = attn.norm_q(query).unflatten(2, (attn.heads, -1))
        key = attn.norm_k(key).unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))
        if rotary_emb is not None:
            query = _apply_rotary(query, *rotary_emb)
            key = _apply_rotary(key, *rotary_emb)
        key, value = context.cache.update(
            context.branch,
            self.layer,
            key,
            value,
            bounds=context.bounds,
            tokens=context.tokens,
            indices=context.indices,
        )
        key, value = all_gather_sequence_pair(key, value, context.cp_group)
        output = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=None,
        )
        output = output.flatten(2, 3).to(query.dtype)
        return attn.to_out[1](attn.to_out[0](output))


def configure_wan_pipeline(model, topology: PipelineTopology) -> None:
    if getattr(model.config, "image_dim", None) is not None:
        raise NotImplementedError("pipeline parallelism currently supports Wan 2.1 T2V")
    existing = getattr(model, "pipeline_topology", None)
    if existing is not None:
        if existing != topology:
            raise ValueError(
                "a partitioned Wan transformer cannot change pipeline topology"
            )
        return
    begin, end = partition_bounds(len(model.blocks), topology.degree)[topology.stage]
    # Keep global block indices in state_dict. Identity placeholders allocate
    # no weights, and the forward visits only this stage's real blocks.
    for index in range(len(model.blocks)):
        if begin <= index < end:
            model.blocks[index].attn1.set_processor(WanFppAttnProcessor(index))
        else:
            model.blocks[index] = nn.Identity()
    model.pipeline_topology = topology
    model.pipeline_layer_range = (begin, end)


def load_wan_pipeline_transformer(
    cls: type,
    pretrained_model_name_or_path: str | Path,
    *,
    topology: PipelineTopology,
    subfolder: str | None = "transformer",
    torch_dtype: torch.dtype | None = None,
    **kwargs: Any,
):
    kwargs.pop("local_files_only", None)
    if kwargs:
        raise TypeError(
            "pipeline-parallel loading does not accept " + ", ".join(sorted(kwargs))
        )
    root = Path(pretrained_model_name_or_path)
    if subfolder:
        root /= subfolder
    if not root.is_dir():
        raise ValueError(
            f"Wan pipeline parallelism needs a local Diffusers checkpoint: {root}"
        )
    config = cls.load_config(root)
    with init_empty_weights(include_buffers=False):
        model = cls.from_config(config)
    if torch_dtype is not None:
        # Match Diffusers' mixed-precision checkpoint loading. Wan keeps time
        # embeddings and modulation tables in FP32. Nonpersistent RoPE buffers
        # retain their construction precision instead of being cast to BF16.
        keep_fp32 = model._keep_in_fp32_modules or []
        for name, parameter in list(model.named_parameters()):
            dtype = (
                torch.float32
                if any(part in name.split(".") for part in keep_fp32)
                else torch_dtype
            )
            module_name, _, leaf = name.rpartition(".")
            setattr(
                model.get_submodule(module_name),
                leaf,
                nn.Parameter(parameter.to(dtype=dtype), requires_grad=False),
            )
    configure_wan_pipeline(model, topology)
    begin, end = model.pipeline_layer_range

    def skip_remote_block(name: str) -> bool:
        parts = name.split(".")
        return (
            len(parts) > 2
            and parts[0] == "blocks"
            and parts[1].isdigit()
            and 0 <= int(parts[1]) < len(model.blocks)
            and not begin <= int(parts[1]) < end
        )

    # The shared loader reads safetensors lazily, checks shapes/missing keys,
    # and materializes only this stage's blocks plus replicated conditioning.
    load_tensor_parallel_checkpoint(
        model, root, device="cpu", skip_checkpoint_parameter=skip_remote_block
    )
    return model.eval().requires_grad_(False)


@torch.no_grad()
def wan_pipeline_forward(
    model,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    *,
    state: FppState,
    branch: str,
    step_index: int,
    full_step: bool,
) -> torch.Tensor:
    """Fill and drain a token pipeline, returning a full prediction on all ranks.

    A logical denoise step is the synchronization boundary. The scheduler is
    called once *after* every patch and both CFG branches have completed, so
    Diffusers multistep history needs no speculative mutation or rollback.
    """
    topology = model.pipeline_topology
    batch, _, frames, height, width = hidden_states.shape
    p_t, p_h, p_w = model.config.patch_size
    if frames % p_t or height % p_h or width % p_w or timestep.ndim != 1:
        raise ValueError(
            "Wan FPP requires patch-aligned latents and one timestep per batch item"
        )
    post_shape = (frames // p_t, height // p_h, width // p_w)
    token_count = post_shape[0] * post_shape[1] * post_shape[2]
    bounds = (
        ((0, token_count),)
        if full_step
        else partition_bounds(token_count, state.config.patches)
    )
    order = (0,) if full_step else state.config.order(step_index)
    rotary = model.rope(hidden_states)
    temb, timestep_proj, context, _ = model.condition_embedder(
        timestep, encoder_hidden_states, None, timestep_seq_len=None
    )
    timestep_proj = timestep_proj.unflatten(1, (6, -1))
    embedded = (
        model.patch_embedding(hidden_states).flatten(2).transpose(1, 2)
        if topology.first
        else None
    )
    dtype, device = hidden_states.dtype, hidden_states.device
    inner_dim = model.config.num_attention_heads * model.config.attention_head_dim
    output_channels = model.config.out_channels or model.config.in_channels
    projection_width = p_t * p_h * p_w * output_channels
    projected = (
        torch.empty(batch, token_count, projection_width, dtype=dtype, device=device)
        if topology.last
        else None
    )
    transport = PipelineTransport(topology)
    begin, end = model.pipeline_layer_range
    for patch in order:
        start, stop = bounds[patch]
        tokens = (
            embedded[:, start:stop].contiguous()
            if embedded is not None
            else transport.receive(
                (batch, stop - start, inner_dim), dtype=dtype, device=device
            )
        )
        patch_rotary = tuple(value[:, start:stop].contiguous() for value in rotary)
        with fpp_attention_context(
            FppAttentionContext(
                state.cache, branch, None if full_step else (start, stop), token_count
            )
        ):
            for block in model.blocks[begin:end]:
                tokens = block(tokens, context, timestep_proj, patch_rotary)
        if topology.last:
            shift, scale = (
                model.scale_shift_table.to(temb.device) + temb.unsqueeze(1)
            ).chunk(2, dim=1)
            tokens = (model.norm_out(tokens.float()) * (1 + scale) + shift).type_as(
                tokens
            )
            projected[:, start:stop] = model.proj_out(tokens)
        else:
            transport.send(tokens)
    transport.drain()

    if projected is not None:
        output = projected.reshape(batch, *post_shape, p_t, p_h, p_w, -1)
        output = output.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = output.flatten(6, 7).flatten(4, 5).flatten(2, 3).contiguous()
    else:
        output = torch.empty(
            batch, output_channels, frames, height, width, dtype=dtype, device=device
        )
    if topology.degree > 1:
        dist.broadcast(output, src=topology.ranks[-1], group=topology.process_group)
    return output
