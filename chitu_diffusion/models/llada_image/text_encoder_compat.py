from __future__ import annotations

import importlib
import importlib.machinery
import importlib.metadata
import sys
import threading
import types
import warnings
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

__all__ = [
    "LLaDATextFrontendCompatibilityStatus",
    "apply_rotary_emb",
    "flash_attn_func",
    "flash_attn_varlen_func",
    "fused_moe_forward",
    "index_first_axis",
    "llada_text_frontend_compatibility",
    "pad_input",
    "unpad_input",
]


_FALLBACK_FLASH_ATTN_VERSION = "2.8.3"
_MODULE_LOCK = threading.RLock()


@dataclass(frozen=True)
class LLaDATextFrontendCompatibilityStatus:
    """Compatibility choices active while the LLaDA remote code is imported."""

    flash_attention_fallback: bool
    veomni_fallback: bool
    flash_attention_error: str | None = None
    veomni_error: str | None = None

    @property
    def fallback_active(self) -> bool:
        return self.flash_attention_fallback or self.veomni_fallback


def _validate_attention_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> None:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError(
            "flash attention inputs must have shape [batch, sequence, heads, head_dim]"
        )
    if query.shape[0] != key.shape[0] or key.shape[0] != value.shape[0]:
        raise ValueError("query, key, and value must have the same batch size")
    if key.shape[1] != value.shape[1]:
        raise ValueError("key and value must have the same sequence length")
    if key.shape[2] != value.shape[2]:
        raise ValueError("key and value must have the same number of heads")
    if query.shape[3] != key.shape[3] or key.shape[3] != value.shape[3]:
        raise ValueError("query, key, and value must have the same head dimension")
    if query.shape[2] % key.shape[2] != 0:
        raise ValueError(
            "the query head count must be divisible by the key/value head count"
        )


def _bottom_right_causal_mask(
    query_length: int,
    key_length: int,
    device: torch.device,
) -> torch.Tensor:
    query_positions = (
        torch.arange(query_length, device=device) + key_length - query_length
    )
    key_positions = torch.arange(key_length, device=device)
    return key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)


def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    _validate_attention_inputs(query, key, value)
    attention_mask = None
    use_sdpa_causal = causal
    if causal and query.shape[1] != key.shape[1]:
        attention_mask = _bottom_right_causal_mask(
            query.shape[1], key.shape[1], query.device
        )
        use_sdpa_causal = False
    output = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=attention_mask,
        dropout_p=dropout_p,
        is_causal=use_sdpa_causal,
        scale=softmax_scale,
        enable_gqa=query.shape[2] != key.shape[2],
    )
    return output.transpose(1, 2).contiguous()


def _validate_flash_options(
    *,
    window_size: tuple[int, int],
    alibi_slopes: torch.Tensor | None,
    return_attn_probs: bool,
    softcap: float,
    extra_options: Mapping[str, object],
) -> None:
    if tuple(window_size) != (-1, -1):
        raise NotImplementedError(
            "the LLaDA fallback does not implement local-window attention"
        )
    if alibi_slopes is not None:
        raise NotImplementedError("the LLaDA fallback does not implement ALiBi")
    if return_attn_probs:
        raise NotImplementedError(
            "the LLaDA fallback does not return attention probabilities"
        )
    if softcap != 0.0:
        raise NotImplementedError(
            "the LLaDA fallback does not implement attention soft-capping"
        )
    if extra_options:
        names = ", ".join(sorted(extra_options))
        raise TypeError(f"unsupported flash attention options: {names}")


def flash_attn_func(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    alibi_slopes: torch.Tensor | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    softcap: float = 0.0,
    **extra_options: object,
) -> torch.Tensor:
    """PyTorch SDPA implementation of the dense FlashAttention API used by LLaDA."""

    del deterministic
    _validate_flash_options(
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        return_attn_probs=return_attn_probs,
        softcap=softcap,
        extra_options=extra_options,
    )
    return _attention(
        query,
        key,
        value,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
    )


def _sequence_boundaries(
    cumulative_lengths: torch.Tensor, total_length: int
) -> list[int]:
    if cumulative_lengths.ndim != 1 or cumulative_lengths.numel() == 0:
        raise ValueError("cumulative sequence lengths must be a non-empty 1D tensor")
    boundaries = (
        cumulative_lengths.detach().to(device="cpu", dtype=torch.int64).tolist()
    )
    if boundaries[0] != 0 or boundaries[-1] != total_length:
        raise ValueError(
            "cumulative sequence lengths must start at zero and end at the packed length"
        )
    if any(stop < start for start, stop in zip(boundaries, boundaries[1:])):
        raise ValueError("cumulative sequence lengths must be nondecreasing")
    return boundaries


def flash_attn_varlen_func(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    alibi_slopes: torch.Tensor | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    softcap: float = 0.0,
    **extra_options: object,
) -> torch.Tensor:
    """PyTorch SDPA implementation of the packed FlashAttention API used by LLaDA."""

    del deterministic
    _validate_flash_options(
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        return_attn_probs=return_attn_probs,
        softcap=softcap,
        extra_options=extra_options,
    )
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError(
            "packed flash attention inputs must have shape [tokens, heads, head_dim]"
        )
    query_boundaries = _sequence_boundaries(cu_seqlens_q, query.shape[0])
    key_boundaries = _sequence_boundaries(cu_seqlens_k, key.shape[0])
    if len(query_boundaries) != len(key_boundaries):
        raise ValueError(
            "query and key cumulative lengths must describe the same batch size"
        )

    outputs: list[torch.Tensor] = []
    for q_start, q_stop, k_start, k_stop in zip(
        query_boundaries,
        query_boundaries[1:],
        key_boundaries,
        key_boundaries[1:],
    ):
        query_length = q_stop - q_start
        key_length = k_stop - k_start
        if query_length > max_seqlen_q or key_length > max_seqlen_k:
            raise ValueError("packed sequence length exceeds the declared maximum")
        if query_length == 0:
            outputs.append(query[q_start:q_stop])
            continue
        if key_length == 0:
            raise ValueError(
                "a non-empty query sequence cannot attend to an empty key sequence"
            )
        outputs.append(
            _attention(
                query[q_start:q_stop].unsqueeze(0),
                key[k_start:k_stop].unsqueeze(0),
                value[k_start:k_stop].unsqueeze(0),
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
            ).squeeze(0)
        )
    if not outputs:
        return query.new_empty(query.shape)
    return torch.cat(outputs, dim=0)


def index_first_axis(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return values.index_select(0, indices.to(device=values.device, dtype=torch.long))


def pad_input(
    hidden_states: torch.Tensor,
    indices: torch.Tensor,
    batch: int,
    seqlen: int,
) -> torch.Tensor:
    if hidden_states.shape[0] != indices.numel():
        raise ValueError(
            "the number of unpadded states must match the number of indices"
        )
    output = hidden_states.new_zeros((batch * seqlen, *hidden_states.shape[1:]))
    output.index_copy_(
        0, indices.to(device=output.device, dtype=torch.long), hidden_states
    )
    return output.view(batch, seqlen, *hidden_states.shape[1:])


def unpad_input(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    unused_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
    if hidden_states.ndim < 2 or attention_mask.shape != hidden_states.shape[:2]:
        raise ValueError("attention_mask must match the batch and sequence dimensions")
    retained_mask = attention_mask.to(dtype=torch.bool)
    if unused_mask is not None:
        if unused_mask.shape != attention_mask.shape:
            raise ValueError("unused_mask must have the same shape as attention_mask")
        retained_mask = retained_mask | unused_mask.to(dtype=torch.bool)

    sequence_lengths = retained_mask.sum(dim=-1, dtype=torch.int32)
    used_lengths = attention_mask.to(dtype=torch.bool).sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(retained_mask.flatten(), as_tuple=False).flatten()
    cumulative_lengths = torch.cat(
        [
            sequence_lengths.new_zeros(1),
            torch.cumsum(sequence_lengths, dim=0, dtype=torch.int32),
        ]
    )
    flat_states = hidden_states.reshape(-1, *hidden_states.shape[2:])
    max_seqlen = int(sequence_lengths.max().item()) if sequence_lengths.numel() else 0
    return (
        flat_states[indices],
        indices,
        cumulative_lengths,
        max_seqlen,
        used_lengths,
    )


def _rotate_half(values: torch.Tensor, interleaved: bool) -> torch.Tensor:
    if interleaved:
        even = values[..., ::2]
        odd = values[..., 1::2]
        return torch.stack((-odd, even), dim=-1).flatten(-2)
    first, second = values.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _normalize_offsets(
    offsets: int | torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(offsets, torch.Tensor):
        flat_offsets = offsets.to(device=device, dtype=torch.long).flatten()
        if flat_offsets.numel() != batch_size:
            raise ValueError("per-sample rotary offsets must match the batch size")
        return flat_offsets
    return torch.full((batch_size,), offsets, device=device, dtype=torch.long)


def _select_dense_rotary_table(
    table: torch.Tensor,
    *,
    batch_size: int,
    sequence_length: int,
    offsets: int | torch.Tensor,
) -> torch.Tensor:
    batch_offsets = _normalize_offsets(
        offsets, batch_size=batch_size, device=table.device
    )
    positions = batch_offsets[:, None] + torch.arange(
        sequence_length, device=table.device
    )
    if table.ndim == 2:
        return table[positions]
    if table.ndim == 3 and table.shape[0] in (1, batch_size):
        expanded = table.expand(batch_size, -1, -1)
        batch_indices = torch.arange(batch_size, device=table.device)[:, None]
        return expanded[batch_indices, positions]
    raise ValueError(
        "rotary tables must have shape [sequence, dim] or [batch, sequence, dim]"
    )


def _select_packed_rotary_table(
    table: torch.Tensor,
    cumulative_lengths: torch.Tensor,
    offsets: int | torch.Tensor,
    total_length: int,
) -> torch.Tensor:
    boundaries = _sequence_boundaries(cumulative_lengths, total_length)
    batch_size = len(boundaries) - 1
    batch_offsets = _normalize_offsets(
        offsets, batch_size=batch_size, device=table.device
    )
    positions = torch.cat(
        [
            torch.arange(stop - start, device=table.device) + batch_offsets[batch_index]
            for batch_index, (start, stop) in enumerate(zip(boundaries, boundaries[1:]))
        ]
    )
    if table.ndim == 2:
        return table[positions]
    if table.ndim == 3 and table.shape[0] in (1, batch_size):
        expanded = table.expand(batch_size, -1, -1)
        batch_indices = torch.cat(
            [
                torch.full(
                    (stop - start,),
                    batch_index,
                    device=table.device,
                    dtype=torch.long,
                )
                for batch_index, (start, stop) in enumerate(
                    zip(boundaries, boundaries[1:])
                )
            ]
        )
        return expanded[batch_indices, positions]
    raise ValueError(
        "rotary tables must have shape [sequence, dim] or [batch, sequence, dim]"
    )


def apply_rotary_emb(
    values: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
    inplace: bool = False,
    seqlen_offsets: int | torch.Tensor = 0,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
    conjugate: bool = False,
) -> torch.Tensor:
    """Apply FlashAttention-compatible rotary embeddings with PyTorch operations."""

    del max_seqlen
    if cos.shape != sin.shape:
        raise ValueError("cos and sin rotary tables must have identical shapes")
    if values.ndim == 4:
        if cu_seqlens is not None:
            raise ValueError("cu_seqlens is only valid for packed rank-3 inputs")
        selected_cos = _select_dense_rotary_table(
            cos,
            batch_size=values.shape[0],
            sequence_length=values.shape[1],
            offsets=seqlen_offsets,
        ).unsqueeze(-2)
        selected_sin = _select_dense_rotary_table(
            sin,
            batch_size=values.shape[0],
            sequence_length=values.shape[1],
            offsets=seqlen_offsets,
        ).unsqueeze(-2)
    elif values.ndim == 3:
        if cu_seqlens is None:
            selected_cos = (
                _select_dense_rotary_table(
                    cos,
                    batch_size=1,
                    sequence_length=values.shape[0],
                    offsets=seqlen_offsets,
                )
                .squeeze(0)
                .unsqueeze(-2)
            )
            selected_sin = (
                _select_dense_rotary_table(
                    sin,
                    batch_size=1,
                    sequence_length=values.shape[0],
                    offsets=seqlen_offsets,
                )
                .squeeze(0)
                .unsqueeze(-2)
            )
        else:
            selected_cos = _select_packed_rotary_table(
                cos, cu_seqlens, seqlen_offsets, values.shape[0]
            ).unsqueeze(-2)
            selected_sin = _select_packed_rotary_table(
                sin, cu_seqlens, seqlen_offsets, values.shape[0]
            ).unsqueeze(-2)
    else:
        raise ValueError(
            "rotary input must have shape [batch, sequence, heads, dim] or "
            "[tokens, heads, dim]"
        )

    rotary_dim = selected_cos.shape[-1] * 2
    if rotary_dim > values.shape[-1]:
        raise ValueError("the rotary dimension cannot exceed the input head dimension")
    rotary = values[..., :rotary_dim]
    if interleaved:
        expanded_cos = torch.repeat_interleave(selected_cos, 2, dim=-1)
        expanded_sin = torch.repeat_interleave(selected_sin, 2, dim=-1)
    else:
        expanded_cos = torch.cat([selected_cos, selected_cos], dim=-1)
        expanded_sin = torch.cat([selected_sin, selected_sin], dim=-1)
    if conjugate:
        expanded_sin = -expanded_sin
    expanded_cos = expanded_cos.to(device=values.device, dtype=values.dtype)
    expanded_sin = expanded_sin.to(device=values.device, dtype=values.dtype)
    result = torch.cat(
        [
            rotary * expanded_cos + _rotate_half(rotary, interleaved) * expanded_sin,
            values[..., rotary_dim:],
        ],
        dim=-1,
    )
    if inplace:
        values.copy_(result)
        return values
    return result


def fused_moe_forward(
    module: nn.Module,
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_weight: torch.Tensor,
    fc1_2_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
) -> torch.Tensor:
    """Evaluate routed SwiGLU experts without the optional VeOmni extension."""

    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if any(
        weight.shape[0] != num_experts
        for weight in (fc1_1_weight, fc1_2_weight, fc2_weight)
    ):
        raise ValueError("all expert weight tensors must contain num_experts experts")
    original_shape = hidden_states.shape
    flat_hidden = hidden_states.reshape(-1, original_shape[-1])
    flat_experts = selected_experts.reshape(flat_hidden.shape[0], -1).to(
        dtype=torch.long
    )
    flat_routing_weights = routing_weights.reshape(flat_hidden.shape[0], -1)
    if flat_experts.shape != flat_routing_weights.shape:
        raise ValueError(
            "selected experts and routing weights must have the same shape"
        )
    if flat_experts.numel() and (
        flat_experts.min() < 0 or flat_experts.max() >= num_experts
    ):
        raise ValueError("selected expert indices are out of range")

    activation = getattr(module, "act_fn", F.silu)
    if not callable(activation):
        raise TypeError("module.act_fn must be callable")
    output = torch.zeros_like(flat_hidden)
    for expert_index in range(num_experts):
        token_indices, route_indices = torch.where(flat_experts == expert_index)
        if token_indices.numel() == 0:
            continue
        expert_input = flat_hidden[token_indices]
        gate = F.linear(expert_input, fc1_1_weight[expert_index])
        up = F.linear(expert_input, fc1_2_weight[expert_index])
        expert_output = F.linear(activation(gate) * up, fc2_weight[expert_index])
        route_weight = flat_routing_weights[token_indices, route_indices, None]
        output.index_add_(
            0,
            token_indices,
            expert_output * route_weight.to(expert_output.dtype),
        )
    return output.view(original_shape)


def _module(name: str, *, package: bool = False) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(
        name, loader=None, is_package=package
    )
    module.__package__ = name if package else name.rpartition(".")[0]
    if package:
        module.__path__ = []
    module.__chitu_llada_fallback__ = True
    return module


def _module_snapshot(root: str) -> dict[str, types.ModuleType]:
    return {
        name: module
        for name, module in sys.modules.items()
        if name == root or name.startswith(f"{root}.")
    }


def _remove_modules(root: str) -> None:
    for name in tuple(sys.modules):
        if name == root or name.startswith(f"{root}."):
            sys.modules.pop(name, None)


def _restore_modules(root: str, snapshot: Mapping[str, types.ModuleType]) -> None:
    _remove_modules(root)
    sys.modules.update(snapshot)


def _probe_flash_attention() -> str | None:
    try:
        flash = importlib.import_module("flash_attn")
        if getattr(flash, "__chitu_llada_fallback__", False):
            return "a fallback flash_attn module is already installed"
        bert_padding = importlib.import_module("flash_attn.bert_padding")
        rotary = importlib.import_module("flash_attn.layers.rotary")
        required = (
            getattr(flash, "flash_attn_func"),
            getattr(flash, "flash_attn_varlen_func"),
            getattr(bert_padding, "index_first_axis"),
            getattr(bert_padding, "pad_input"),
            getattr(bert_padding, "unpad_input"),
            getattr(rotary, "apply_rotary_emb"),
        )
        if not all(callable(item) for item in required):
            return (
                "flash_attn does not expose all functions required by the "
                "LLaDA frontend"
            )
        importlib.metadata.version("flash_attn")
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    return None


def _probe_veomni() -> str | None:
    try:
        veomni = importlib.import_module("veomni")
        if getattr(veomni, "__chitu_llada_fallback__", False):
            return "a fallback veomni module is already installed"
        ops = importlib.import_module("veomni.ops")
        if not callable(getattr(ops, "fused_moe_forward")):
            return "veomni.ops.fused_moe_forward is not callable"
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    return None


def _install_flash_attention_fallback() -> None:
    flash = _module("flash_attn", package=True)
    flash.__version__ = _FALLBACK_FLASH_ATTN_VERSION
    flash.flash_attn_func = flash_attn_func
    flash.flash_attn_varlen_func = flash_attn_varlen_func

    interface = _module("flash_attn.flash_attn_interface")
    interface.flash_attn_func = flash_attn_func
    interface.flash_attn_varlen_func = flash_attn_varlen_func
    bert_padding = _module("flash_attn.bert_padding")
    bert_padding.index_first_axis = index_first_axis
    bert_padding.pad_input = pad_input
    bert_padding.unpad_input = unpad_input
    layers = _module("flash_attn.layers", package=True)
    rotary = _module("flash_attn.layers.rotary")
    rotary.apply_rotary_emb = apply_rotary_emb

    flash.flash_attn_interface = interface
    flash.bert_padding = bert_padding
    flash.layers = layers
    layers.rotary = rotary
    sys.modules.update(
        {
            "flash_attn": flash,
            "flash_attn.flash_attn_interface": interface,
            "flash_attn.bert_padding": bert_padding,
            "flash_attn.layers": layers,
            "flash_attn.layers.rotary": rotary,
        }
    )


def _install_veomni_fallback() -> None:
    veomni = _module("veomni", package=True)
    ops = _module("veomni.ops")
    ops.fused_moe_forward = fused_moe_forward
    veomni.ops = ops
    sys.modules.update({"veomni": veomni, "veomni.ops": ops})


def _fallback_version(
    original_version: Callable[[str], str],
) -> Callable[[str], str]:
    def version(distribution_name: str) -> str:
        normalized_name = distribution_name.lower().replace("-", "_")
        if normalized_name == "flash_attn":
            return _FALLBACK_FLASH_ATTN_VERSION
        return original_version(distribution_name)

    return version


@contextmanager
def llada_text_frontend_compatibility() -> Iterator[
    LLaDATextFrontendCompatibilityStatus
]:
    """Temporarily provide extension fallbacks while importing LLaDA remote code.

    Native FlashAttention and VeOmni modules stay untouched when their import-time
    ABI probes succeed. Failed modules and the FlashAttention distribution version
    lookup are replaced only for the duration of this context.
    """

    with _MODULE_LOCK:
        flash_snapshot = _module_snapshot("flash_attn")
        veomni_snapshot = _module_snapshot("veomni")
        original_metadata_version = importlib.metadata.version
        flash_error = _probe_flash_attention()
        veomni_error = _probe_veomni()
        flash_fallback = flash_error is not None
        veomni_fallback = veomni_error is not None
        status = LLaDATextFrontendCompatibilityStatus(
            flash_attention_fallback=flash_fallback,
            veomni_fallback=veomni_fallback,
            flash_attention_error=flash_error,
            veomni_error=veomni_error,
        )

        try:
            if flash_fallback:
                _remove_modules("flash_attn")
                _install_flash_attention_fallback()
                importlib.metadata.version = _fallback_version(
                    original_metadata_version
                )
            if veomni_fallback:
                _remove_modules("veomni")
                _install_veomni_fallback()
            if status.fallback_active:
                fallbacks = []
                if flash_fallback:
                    fallbacks.append(f"flash_attn ({flash_error})")
                if veomni_fallback:
                    fallbacks.append(f"veomni ({veomni_error})")
                warnings.warn(
                    "LLaDA text frontend is using mathematically equivalent "
                    "PyTorch fallbacks for "
                    + ", ".join(fallbacks)
                    + "; optimized extension kernels are unavailable.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            yield status
        finally:
            if flash_fallback:
                importlib.metadata.version = original_metadata_version
                _restore_modules("flash_attn", flash_snapshot)
            if veomni_fallback:
                _restore_modules("veomni", veomni_snapshot)
