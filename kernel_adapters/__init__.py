"""Compatibility adapters for upstream GPU kernels.

Implementations and algorithms are adapted from reviewed upstream projects,
including SGLang, vLLM, FastVideo, FlashAttention, and Hugging Face kernels.
Individual modules retain their SPDX identifiers, license terms, and precise
source provenance.
"""

from .attention import (
    flash_attention3_available,
    flash_attention3_forward,
    flash_attention3_provider,
)
from .attention_state_merge import (
    merge_attention_states_reference,
    merge_mode,
    triton_merge_available,
    try_merge_attention_states,
)
from .zimage_native_norm import (
    rmsnorm_scale,
    rmsnorm_tanh_residual,
    triton_available,
)
from .qwen_image_modulation import (
    layernorm_scale_shift_gate_select01,
    residual_layernorm_scale_shift_gate_select01,
)
from .silu_and_mul import can_use_silu_and_mul, silu_and_mul

__all__ = [
    "flash_attention3_available",
    "flash_attention3_forward",
    "flash_attention3_provider",
    "merge_attention_states_reference",
    "merge_mode",
    "triton_merge_available",
    "try_merge_attention_states",
    "rmsnorm_scale",
    "rmsnorm_tanh_residual",
    "triton_available",
    "layernorm_scale_shift_gate_select01",
    "residual_layernorm_scale_shift_gate_select01",
    "can_use_silu_and_mul",
    "silu_and_mul",
]
