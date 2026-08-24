from __future__ import annotations

import torch
from torch import nn

from chitu_diffusion.parallel.attention_backend import (
    VarlenAttentionBackend,
    create_varlen_attention_backend,
)
from chitu_diffusion.parallel.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from chitu_diffusion.parallel.packed_usp import (
    all_to_all_packed_output,
    all_to_all_packed_qkv,
)
from chitu_diffusion.parallel.tensor_parallel import get_tp_world_size
from chitu_diffusion.parallel.topology import UspTopology

from .config import MiniMaxH3DiTConfig
from .rope import apply_rope


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
        if ulysses_topology is not None and ulysses_topology.ulysses_degree > 1:
            query, key, value = all_to_all_packed_qkv(
                query, key, value, topology=ulysses_topology
            )
        output = self.backend.forward_varlen(
            query,
            key,
            value,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        if ulysses_topology is not None and ulysses_topology.ulysses_degree > 1:
            output = all_to_all_packed_output(
                output, topology=ulysses_topology
            )
        output = output.reshape(token_count, self.local_inner_dim)
        output, _ = self.out_proj(output)
        return output

