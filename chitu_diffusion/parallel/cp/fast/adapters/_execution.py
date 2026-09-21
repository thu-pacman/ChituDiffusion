"""Shared prepared AGKV schedule; model math is supplied by adapter hooks.

This module owns segmented CUDA graphs and orders KV production, CE transport,
Q projection, attention and output projection. It has no model imports.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict

import torch
import triton

from ..attention.planner import _execute, prepare_attention
from ..projection.kv import PersistentKVProjection


def _capture(call):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = call()
    graph.replay()
    return graph, output


class PreparedAgkv(ABC):
    """Fixed-buffer AGKV execution shared by model-specific subclasses."""

    _supports_local_key_postprocess = True

    def __init__(
        self,
        runtime,
        attn,
        hidden_states,
        rotary_emb,
        *,
        chunk_tokens=None,
        remote_streams=1,
        peer_order="xor",
        attention_backend="planned",
        projection_backend="cublas",
        tag="agkv",
    ):
        if not torch.is_inference_mode_enabled() or attn.training:
            raise ValueError("prepare AGKV in inference_mode with eval weights")
        if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
            raise ValueError("prepared AGKV requires B1 sequence shards")
        if not hidden_states.is_contiguous():
            raise ValueError("prepared AGKV requires contiguous hidden states")
        if attention_backend not in ("planned", "sdpa"):
            raise ValueError("attention backend must be planned or sdpa")
        if chunk_tokens is None:
            local_tokens = hidden_states.shape[1]
            chunk_tokens = (
                256
                if projection_backend == "persistent"
                else triton.cdiv(local_tokens, 128) * 128
            )
        self.attn, self.hidden_states = attn, hidden_states
        self.rotary_emb, self.runtime = rotary_emb, runtime
        self.closed = False
        self.lane = torch.cuda.current_stream(hidden_states.device)
        self.weight = torch.cat([attn.to_k.weight, attn.to_v.weight]).contiguous()
        if (attn.to_k.bias is None) != (attn.to_v.bias is None):
            raise ValueError("K/V must agree on bias presence")
        self.bias = (
            None
            if attn.to_k.bias is None
            else torch.cat([attn.to_k.bias, attn.to_v.bias]).contiguous()
        )
        width = attn.to_k.out_features
        if attn.to_q.out_features != width or attn.to_v.out_features != width:
            raise ValueError("prepared AGKV requires equal Q/K/V head counts")
        self.projection = PersistentKVProjection(
            runtime,
            hidden_states.flatten(0, 1),
            self.weight,
            heads=attn.heads,
            head_dim=width // attn.heads,
            bias=self.bias,
            chunk_tokens=chunk_tokens,
            remote_streams=remote_streams,
            peer_order=peer_order,
            projection_backend=projection_backend,
            tag=tag,
        )
        self.local_key_postprocess = (
            projection_backend == "cublas" and self._supports_local_key_postprocess
        )
        self.full_rotary = (
            None
            if self.local_key_postprocess
            else self._gather_rotary(rotary_emb, width)
        )
        self.q_graph, self.head = _capture(self._head)
        self.query = self.head[0]
        raw_key, value = self.projection(
            hidden_states.flatten(0, 1),
            self.weight,
            self.bias,
            key_postprocess=(
                self._postprocess_local_key if self.local_key_postprocess else None
            ),
        )
        key, prepared_value = self._key_value(raw_key, value)
        self.attention = prepare_attention(
            self.query, key, prepared_value, cp_degree=self.projection.world
        )
        if attention_backend == "planned" and self.attention.plan.strategy == "sdpa":
            raise RuntimeError(
                "prepared attention planner unavailable; check the reviewed CuTe PYTHONPATH"
            )
        if attention_backend == "sdpa":
            from ..attention import AttentionPlan

            self.attention.plan = AttentionPlan()
        self.plan = self.attention.plan

        def tail():
            key, prepared_value = self._key_value(raw_key, value)
            output = _execute(
                self.plan, self.attention.forward, self.query, key, prepared_value
            )
            return self._finish(output)

        self.tail_graph, self.output = _capture(tail)
        self.metadata = {
            "attention_plan": asdict(self.plan),
            "chunk_tokens": chunk_tokens,
            "remote_streams": remote_streams,
            "peer_order": peer_order,
            "projection_backend": projection_backend,
            "projection_output": "direct symmetric local slot",
            "local_copy_bytes": 0,
            "local_key_postprocess": self.local_key_postprocess,
            "kv_gemm_transport_overlap": projection_backend == "persistent",
            "kv_transport_q_overlap": True,
            "normalization": (
                "K RMSNorm and RoPE in-place before send; Q before attention"
                if self.local_key_postprocess
                else "K RMSNorm and RoPE after gather; Q before attention"
            ),
            "gemm_launches_kv": 1 if projection_backend == "persistent" else 2,
        }

    def _finish(self, output):
        return self.attn.to_out[1](self.attn.to_out[0](output.flatten(2).contiguous()))

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        rotary_emb=None,
        *,
        overlap=True,
        overlap_q=True,
    ):
        if (
            self.closed
            or attn is not self.attn
            or not torch.is_inference_mode_enabled()
        ):
            raise RuntimeError("prepared AGKV requires its live inference slot")
        if encoder_hidden_states is not None or attention_mask is not None:
            raise ValueError("prepared AGKV supports unmasked self-attention only")
        if torch.cuda.current_stream(hidden_states.device) != self.lane:
            raise RuntimeError("prepared AGKV requires its construction stream")
        if (
            hidden_states.shape != self.hidden_states.shape
            or hidden_states.stride() != self.hidden_states.stride()
            or hidden_states.data_ptr() != self.hidden_states.data_ptr()
            or hidden_states.dtype != self.hidden_states.dtype
            or hidden_states.device != self.hidden_states.device
        ):
            raise ValueError("prepare another AGKV slot for different input storage")
        if (
            (rotary_emb is None) != (self.rotary_emb is None)
            or rotary_emb is not None
            and any(
                a.data_ptr() != b.data_ptr() or a.shape != b.shape
                for a, b in zip(rotary_emb, self.rotary_emb, strict=True)
            )
        ):
            raise ValueError("prepare another AGKV slot for different RoPE tables")
        pending = self.projection.begin(
            hidden_states.flatten(0, 1),
            self.weight,
            self.bias,
            overlap=overlap,
            key_postprocess=(
                self._postprocess_local_key if self.local_key_postprocess else None
            ),
        )
        if not overlap_q:
            pending.wait()
        self.q_graph.replay()
        pending.wait()
        self.tail_graph.replay()
        return self.output

    def close(self):
        if self.closed:
            return
        self.projection.close()
        self.closed = True

    @abstractmethod
    def _gather_rotary(self, rotary_emb, width):
        """Prepare the global RoPE layout for post-gather normalization."""

    @abstractmethod
    def _head(self):
        """Compute local queries and optional replicated-text projections."""

    @abstractmethod
    def _postprocess_local_key(self, raw_key):
        """Normalize/rotate the local symmetric key slot before transmission."""

    @abstractmethod
    def _key_value(self, raw_key, value):
        """Produce the model's final KV layout for attention."""
