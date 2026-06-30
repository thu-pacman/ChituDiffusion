from __future__ import annotations

import os
from logging import getLogger
from typing import Optional

import torch
import torch.nn.functional as F

from diffusers.models.attention_processor import Attention

from chitu_diffusion.modules.attention.image_cp_attention import (
    ImageContextParallelAttention,
    resolve_cp_mode,
)

logger = getLogger(__name__)


class ZImageChituAttnProcessor:
    """Z-Image single-stream attention processor backed by Chitu attention.

    Model-specific surface only: QKV projection, Z-Image RoPE, the optional
    attention mask, and the single-stream ``[image, text]`` packing. The actual
    context-parallel attention (replicated text + sharded image, with the
    agkv/ulysses/ring/ring_graph strategies and CP-comm profiling) lives in the
    model-agnostic ``ImageContextParallelAttention`` core, which Qwen-Image can
    reuse as well.
    """

    def __init__(self, attn_backend):
        # The runtime wraps the raw kernel in DiffusionAttention_with_CP when
        # cp_size>1; the CP core unwraps it. Keep the same unwrapped kernel here
        # for the non-CP (mask / single-rank) attention paths below.
        self.attn_backend = getattr(attn_backend, "attn", attn_backend)

        # Optional CP communication profiling (CHITU_Z_IMAGE_CP_PROFILE=1):
        # synchronized CUDA-event timing of every comm/compute op so the adapter
        # can report a comm-vs-compute breakdown. Zero overhead when off.
        self._cp_profile = os.environ.get("CHITU_Z_IMAGE_CP_PROFILE", "0") == "1"
        mode = resolve_cp_mode("CHITU_Z_IMAGE_CP_MODE", "agkv")
        use_cudagraph = os.environ.get("CHITU_Z_IMAGE_RING_CUDAGRAPH", "1") == "1"
        # Ulysses degree for the 2D ``unified`` mode. Must match the framework
        # ``up`` config (parallel.up) so the matching UP sub-group exists; 1 =>
        # pure ring, cp_size => pure ulysses.
        ulysses_size = int(os.environ.get("CHITU_Z_IMAGE_CP_UP", "1"))

        self._cp = ImageContextParallelAttention(
            attn_backend,
            mode=mode,
            profile=self._cp_profile,
            use_cudagraph=use_cudagraph,
            ulysses_size=ulysses_size,
        )
        # Local image-shard length within the packed [image, text] sequence,
        # set when CP is enabled by the adapter's CP-aware transformer forward.
        self._cp_image_len = 0

    # ------------------------------------------------------------------ #
    # Adapter-facing CP surface (delegates to the core)
    # ------------------------------------------------------------------ #
    def enable_cp(self, group, image_len: int) -> None:
        """Activate CP: local hidden_states hold ``image_len`` local image tokens
        (at the front) followed by the full, replicated text tokens."""
        self._cp_image_len = int(image_len)
        self._cp.enable(group)

    def disable_cp(self) -> None:
        self._cp.disable()

    def reset_cp_stats(self) -> None:
        self._cp.reset_stats()

    def cp_stats(self) -> dict:
        return self._cp.stats()

    def _cp_all_gather_seq(self, tensor: torch.Tensor, group, kind: str = "kv") -> torch.Tensor:
        """All-gather a ``[B, S_local, H, D]`` tensor along the sequence dim.

        Used by the adapter for the post-layers output gather (``kind="out"``).
        """
        return self._cp.all_gather_seq(tensor, group, kind=kind)

    # ------------------------------------------------------------------ #
    # Model-specific helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
            freqs_cis = freqs_cis.unsqueeze(2)
            x_out = torch.view_as_real(x * freqs_cis).flatten(3)
            return x_out.type_as(x_in)

    @staticmethod
    def _mask_allows_fast_path(attention_mask: Optional[torch.Tensor]) -> bool:
        if attention_mask is None:
            return True
        # Z-Image uses ragged prompt/image sequences internally. Flash attention is
        # safe only when the packed sequence is already fully valid; otherwise keep
        # the explicit mask and fall back to SDPA for correctness.
        return bool(attention_mask.to(torch.bool).all().item())

    @staticmethod
    def _sdpa_with_mask(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        attn_mask = None
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                attn_mask = attention_mask[:, None, None, :].to(torch.bool)
            elif attention_mask.ndim == 4:
                attn_mask = attention_mask.to(torch.bool)
            else:
                raise ValueError(f"Unsupported Z-Image attention mask shape: {tuple(attention_mask.shape)}")
        out = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        return out.transpose(1, 2).contiguous()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        def _qkv_proj():
            return attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)

        query, key, value = self._cp.time(_qkv_proj, "qkv")

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if freqs_cis is not None:
            query = self._apply_rotary_emb(query, freqs_cis)
            key = self._apply_rotary_emb(key, freqs_cis)

        dtype = query.dtype
        query = query.to(value.dtype).contiguous()
        key = key.to(value.dtype).contiguous()
        value = value.contiguous()

        if self._cp.enabled and self._mask_allows_fast_path(attention_mask):
            # Local Q/K/V are packed [image_chunk, replicated_text]; RoPE is
            # already applied per local position. Split into the image shard and
            # the replicated text, run the model-agnostic CP attention, then
            # reassemble the [image_chunk, text] layout the wrapper expects.
            img_len = self._cp_image_len
            txt_out, img_out = self._cp.attention(
                query[:, img_len:], key[:, img_len:], value[:, img_len:],
                query[:, :img_len], key[:, :img_len], value[:, :img_len],
            )
            hidden_states = torch.cat([img_out, txt_out], dim=1).contiguous()
        elif self._mask_allows_fast_path(attention_mask):
            hidden_states = self._cp.time(
                lambda: self.attn_backend(query, key, value, causal=False)[0], "attn"
            )
        else:
            hidden_states = self._cp.time(
                lambda: self._sdpa_with_mask(query, key, value, attention_mask), "attn"
            )

        hidden_states = hidden_states.flatten(2, 3).to(dtype)
        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:
            output = attn.to_out[1](output)
        return output
