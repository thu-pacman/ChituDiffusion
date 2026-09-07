from __future__ import annotations

import torch
from torch import nn

from chitu_diffusion.parallel.cp.attention_backend import (
    VarlenAttentionBackend,
    create_varlen_attention_backend,
)
from chitu_diffusion.parallel.cp.packed_usp import (
    all_gather_packed_kv,
    all_to_all_packed_output,
    all_to_all_packed_qkv,
)
from chitu_diffusion.parallel.cp.topology import UspTopology
from chitu_diffusion.parallel.tp.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from chitu_diffusion.parallel.tp.topology import get_tp_world_size

from .config import MiniMaxH3DiTConfig
from .rope import apply_rope


def _local_cu_seqlens(
    cu_seqlens: torch.Tensor,
    *,
    rank: int,
    local_tokens: int,
) -> torch.Tensor:
    """Intersect global packed segments with one contiguous equal-size shard."""

    offset = int(rank) * int(local_tokens)
    stop = offset + int(local_tokens)
    starts = cu_seqlens[:-1].clamp(min=offset, max=stop)
    ends = cu_seqlens[1:].clamp(min=offset, max=stop)
    lengths = (ends - starts).clamp_min(0)
    return torch.cat((torch.zeros_like(cu_seqlens[:1]), lengths.cumsum(0)))


class MiniMaxH3Attention(nn.Module):
    def __init__(
        self,
        config: MiniMaxH3DiTConfig,
        *,
        attention_backend: str = "auto",
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        tp_size = get_tp_world_size()
        if config.num_attention_heads % tp_size:
            raise ValueError("attention heads must be divisible by TP degree")
        self.total_num_heads = config.num_attention_heads
        self.num_heads = config.num_attention_heads // tp_size
        self.head_dim = config.attention_head_dim
        self.inner_dim = config.inner_dim
        self.local_inner_dim = self.num_heads * self.head_dim
        self.qkv_proj = MergedColumnParallelLinear(
            config.hidden_size,
            (self.inner_dim, self.inner_dim, self.inner_dim),
            bias=False,
            params_dtype=torch.bfloat16,
            device=device,
        )
        self.qkv_proj.num_attention_heads = self.total_num_heads
        self.qkv_proj.attention_head_dim = self.head_dim
        self.q_norm = nn.RMSNorm(
            self.head_dim,
            eps=config.qk_norm_eps,
            dtype=torch.bfloat16,
            device=device,
        )
        self.k_norm = nn.RMSNorm(
            self.head_dim,
            eps=config.qk_norm_eps,
            dtype=torch.bfloat16,
            device=device,
        )
        self.out_proj = RowParallelLinear(
            self.inner_dim,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
            params_dtype=torch.bfloat16,
            device=device,
        )
        self.backend: VarlenAttentionBackend = create_varlen_attention_backend(
            attention_backend
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        rope_frequencies: torch.Tensor | None,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        ulysses_topology: UspTopology | None = None,
        attention_mode: str = "ulysses",
    ) -> torch.Tensor:
        token_count = hidden_states.shape[0]
        qkv, _ = self.qkv_proj(hidden_states)
        query, key, value = qkv.split(self.local_inner_dim, dim=-1)
        query = query.view(token_count, self.num_heads, self.head_dim)
        key = key.view(token_count, self.num_heads, self.head_dim)
        value = value.view(token_count, self.num_heads, self.head_dim)
        query = self.q_norm(query)
        key = self.k_norm(key)
        if rope_frequencies is not None:
            query = apply_rope(query, rope_frequencies)
            key = apply_rope(key, rope_frequencies)
        parallel = (
            ulysses_topology is not None and ulysses_topology.ulysses_degree > 1
        )
        if ulysses_topology is not None:
            attention_mode = ulysses_topology.attention_mode
        if attention_mode not in {"ulysses", "agkv"}:
            raise ValueError("attention_mode must be 'ulysses' or 'agkv'")
        query_cu = cu_seqlens
        key_cu = None
        query_max_seqlen = max_seqlen
        if parallel and attention_mode == "ulysses":
            query, key, value = all_to_all_packed_qkv(
                query, key, value, topology=ulysses_topology
            )
        elif parallel:
            key, value = all_gather_packed_kv(
                key, value, topology=ulysses_topology
            )
            query_cu = _local_cu_seqlens(
                cu_seqlens,
                rank=ulysses_topology.ulysses_rank,
                local_tokens=token_count,
            )
            query_max_seqlen = min(int(max_seqlen), token_count)
        output = self.backend.forward_varlen(
            query,
            key,
            value,
            cu_seqlens=query_cu,
            max_seqlen=query_max_seqlen,
            cu_seqlens_k=key_cu if key_cu is not None else (
                cu_seqlens if parallel and attention_mode == "agkv" else None
            ),
            max_seqlen_k=max_seqlen if parallel and attention_mode == "agkv" else None,
        )
        if parallel and attention_mode == "ulysses":
            output = all_to_all_packed_output(
                output, topology=ulysses_topology
            )
        output = output.reshape(token_count, self.local_inner_dim)
        output, _ = self.out_proj(output)
        return output

