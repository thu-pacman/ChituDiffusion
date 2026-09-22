"""Prepared Qwen-Image joint attention using the KV projection CE pipeline."""

import statistics
from dataclasses import asdict, dataclass
from weakref import WeakKeyDictionary

import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen

from ..attention.planner import _execute, prepare_attention
from ..projection.kv import PersistentKVProjection
from ..projection.qk_epilogue import joint_qkv_norm_rope, qk_norm_rope_inplace
from ._execution import PreparedAgkv, _capture
from .base import (
    ProcessorExecution,
    UnsupportedFastCP,
    _rope,
    per_head_norms,
    validate_common,
)

_schedule_cache = WeakKeyDictionary()


def qwen_normalize_rotary(attn, tensor, rotary, *, query, text=False):
    norm = (
        (attn.norm_added_q if query else attn.norm_added_k)
        if text
        else (attn.norm_q if query else attn.norm_k)
    )
    result = tensor.unflatten(-1, (attn.heads, -1))
    if norm is not None:
        result = norm(result)
    if rotary is not None:
        result = apply_rotary_emb_qwen(result, rotary, use_real=False)
    return result.to(tensor.dtype).contiguous()


class PreparedQwenImageAgkv(PreparedAgkv):
    """Fixed-storage, all-valid text-prefix joint attention; returns image,text."""

    _supports_local_key_postprocess = True

    def __init__(
        self,
        runtime,
        attn,
        hidden_states,
        encoder_hidden_states,
        image_rotary_emb,
        **kwargs,
    ):
        if (
            encoder_hidden_states.ndim != 3
            or encoder_hidden_states.shape[0] != 1
            or encoder_hidden_states.shape[2] != hidden_states.shape[2]
            or not encoder_hidden_states.is_contiguous()
        ):
            raise ValueError("Qwen AGKV requires contiguous B1 replicated text")
        self.encoder_hidden_states = encoder_hidden_states
        projections = (attn.add_q_proj, attn.add_k_proj, attn.add_v_proj)
        self.text_weight = torch.cat(
            [projection.weight for projection in projections]
        ).contiguous()
        biases = [projection.bias for projection in projections]
        if any(bias is None for bias in biases) != all(bias is None for bias in biases):
            raise ValueError("Qwen text Q/K/V must agree on bias presence")
        self.text_bias = None if biases[0] is None else torch.cat(biases).contiguous()
        super().__init__(runtime, attn, hidden_states, image_rotary_emb, **kwargs)
        self.metadata["joint_tokens"] = encoder_hidden_states.shape[1]
        self.metadata["normalization"] = (
            "local image K and Q/text per-head RMSNorm+RoPE epilogues"
        )
        self.metadata["text_qkv_gemm_launches"] = 1

    def _gather_rotary(self, rotary_emb, width):
        if rotary_emb is None:
            return None
        image, text = rotary_emb
        if image.shape[0] != self.hidden_states.shape[1]:
            raise ValueError("Qwen image RoPE must match the local sequence shard")
        gathered = [torch.empty_like(image) for _ in range(self.projection.world)]
        dist.all_gather(gathered, image.contiguous(), group=self.projection.pg)
        return torch.cat(gathered, dim=0), text

    def _head(self):
        attn = self.attn
        image_rope, text_rope = self.rotary_emb or (None, None)
        iq = qk_norm_rope_inplace(
            attn.to_q(self.hidden_states),
            attn.norm_q,
            image_rope,
            heads=attn.heads,
        )
        text_qkv = F.linear(
            self.encoder_hidden_states, self.text_weight, self.text_bias
        )
        tq, tk, tv = text_qkv.chunk(3, dim=-1)
        tq = qk_norm_rope_inplace(
            tq,
            attn.norm_added_q,
            text_rope,
            heads=attn.heads,
        )
        tk = qk_norm_rope_inplace(
            tk,
            attn.norm_added_k,
            text_rope,
            heads=attn.heads,
        )
        tv = tv.unflatten(-1, (attn.heads, -1))
        return torch.cat([tq, iq], dim=1), tk, tv

    def _postprocess_local_key(self, raw_key):
        image_rope = self.rotary_emb[0] if self.rotary_emb is not None else None
        qk_norm_rope_inplace(
            raw_key,
            self.attn.norm_k,
            image_rope,
            heads=self.attn.heads,
        )

    def _key_value(self, raw_key, value):
        if self.local_key_postprocess:
            key = raw_key
        else:
            image_rope = self.full_rotary[0] if self.full_rotary is not None else None
            key = qwen_normalize_rotary(
                self.attn, raw_key.flatten(2), image_rope, query=False
            )
        return torch.cat([self.head[1], key], dim=1), torch.cat(
            [self.head[2], value], dim=1
        )

    def _finish(self, output):
        text_tokens = self.encoder_hidden_states.shape[1]
        text, image = (
            output[:, :text_tokens].flatten(2),
            output[:, text_tokens:].flatten(2),
        )
        image = self.attn.to_out[0](image.contiguous())
        if len(self.attn.to_out) > 1:
            image = self.attn.to_out[1](image)
        return image, self.attn.to_add_out(text.contiguous())

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        encoder_hidden_states_mask=None,
        attention_mask=None,
        image_rotary_emb=None,
        *,
        overlap=True,
        overlap_q=True,
    ):
        if encoder_hidden_states is None or (
            encoder_hidden_states.data_ptr() != self.encoder_hidden_states.data_ptr()
            or encoder_hidden_states.shape != self.encoder_hidden_states.shape
            or encoder_hidden_states.stride() != self.encoder_hidden_states.stride()
        ):
            raise ValueError(
                "prepare another Qwen AGKV slot for different text storage"
            )
        if encoder_hidden_states_mask is not None or attention_mask is not None:
            raise ValueError(
                "prepared Qwen AGKV requires compact all-valid text without masks"
            )
        return super().__call__(
            attn,
            hidden_states,
            rotary_emb=image_rotary_emb,
            overlap=overlap,
            overlap_q=overlap_q,
        )


class PreparedQwenImageShardedAgkv:
    """Qwen joint attention with both image and text sequence-sharded."""

    def __init__(
        self,
        runtime,
        attn,
        hidden_states,
        encoder_hidden_states,
        image_rotary_emb,
        *,
        remote_streams=1,
        peer_order="xor",
        tag="qwen_sharded_agkv",
    ):
        if not torch.is_inference_mode_enabled() or attn.training:
            raise ValueError("prepare Qwen AGKV in inference mode with eval weights")
        for name, tensor in (
            ("image", hidden_states),
            ("text", encoder_hidden_states),
        ):
            if tensor.ndim != 3 or tensor.shape[0] != 1 or not tensor.is_contiguous():
                raise ValueError(f"Qwen {name} shard must be contiguous B1")
        if hidden_states.shape[2] != encoder_hidden_states.shape[2]:
            raise ValueError("Qwen image and text hidden widths must match")
        self.runtime = runtime
        self.attn = attn
        self.hidden_states = hidden_states
        self.encoder_hidden_states = encoder_hidden_states
        self.rotary_emb = image_rotary_emb
        self.lane = torch.cuda.current_stream(hidden_states.device)
        self.closed = False

        image_layers = (attn.to_k, attn.to_v)
        text_layers = (attn.add_k_proj, attn.add_v_proj)
        image_weight, image_bias = self._pack(image_layers)
        text_weight, text_bias = self._pack(text_layers)
        width = attn.to_k.out_features
        projection_options = {
            "heads": attn.heads,
            "head_dim": width // attn.heads,
            "remote_streams": remote_streams,
            "peer_order": peer_order,
            "projection_backend": "cublas",
        }
        self.image_projection = PersistentKVProjection(
            runtime,
            hidden_states.flatten(0, 1),
            image_weight,
            bias=image_bias,
            tag=tag + "_image",
            **projection_options,
        )
        self.text_projection = PersistentKVProjection(
            runtime,
            encoder_hidden_states.flatten(0, 1),
            text_weight,
            bias=text_bias,
            tag=tag + "_text",
            **projection_options,
        )
        self.image_weight, self.image_bias = image_weight, image_bias
        self.text_weight, self.text_bias = text_weight, text_bias

        self.q_graph, self.query = _capture(self._head)
        image_key, image_value = self.image_projection(
            hidden_states.flatten(0, 1),
            image_weight,
            image_bias,
            key_postprocess=self._postprocess_image_key,
        )
        text_key, text_value = self.text_projection(
            encoder_hidden_states.flatten(0, 1),
            text_weight,
            text_bias,
            key_postprocess=self._postprocess_text_key,
        )
        key = torch.cat([text_key, image_key], dim=1)
        value = torch.cat([text_value, image_value], dim=1)
        self.attention = prepare_attention(
            self.query,
            key,
            value,
            cp_degree=self.image_projection.world,
        )
        self.plan = self.attention.plan
        if self.plan.strategy == "sdpa":
            raise RuntimeError("Qwen sharded AGKV requires the CuTe attention planner")

        def tail():
            key = torch.cat([text_key, image_key], dim=1)
            value = torch.cat([text_value, image_value], dim=1)
            output = _execute(self.plan, self.attention.forward, self.query, key, value)
            return self._finish(output)

        self.tail_graph, self.output = _capture(tail)
        self.metadata = {
            "attention_plan": asdict(self.plan),
            "partition": "image and text sequence-sharded",
            "projection_output": "two direct symmetric local slots",
            "local_key_postprocess": True,
            "kv_transport_q_overlap": True,
            "normalization": "local per-head RMSNorm+RoPE before CE send",
            "gemm_launches_kv": 4,
            "gemm_launches_q": 2,
        }

    @staticmethod
    def _pack(layers):
        weight = torch.cat([layer.weight for layer in layers]).contiguous()
        biases = [layer.bias for layer in layers]
        if any(bias is None for bias in biases) != all(bias is None for bias in biases):
            raise ValueError("Qwen K/V layers must agree on bias presence")
        bias = None if biases[0] is None else torch.cat(biases).contiguous()
        return weight, bias

    def _head(self):
        image_rope, text_rope = self.rotary_emb
        image_query = qk_norm_rope_inplace(
            self.attn.to_q(self.hidden_states),
            self.attn.norm_q,
            image_rope,
            heads=self.attn.heads,
        )
        text_query = qk_norm_rope_inplace(
            self.attn.add_q_proj(self.encoder_hidden_states),
            self.attn.norm_added_q,
            text_rope,
            heads=self.attn.heads,
        )
        return torch.cat([text_query, image_query], dim=1)

    def _postprocess_image_key(self, raw_key):
        qk_norm_rope_inplace(
            raw_key,
            self.attn.norm_k,
            self.rotary_emb[0],
            heads=self.attn.heads,
        )

    def _postprocess_text_key(self, raw_key):
        qk_norm_rope_inplace(
            raw_key,
            self.attn.norm_added_k,
            self.rotary_emb[1],
            heads=self.attn.heads,
        )

    def _finish(self, output):
        text_tokens = self.encoder_hidden_states.shape[1]
        text = output[:, :text_tokens].flatten(2)
        image = output[:, text_tokens:].flatten(2)
        image = self.attn.to_out[0](image.contiguous())
        if len(self.attn.to_out) > 1:
            image = self.attn.to_out[1](image)
        return image, self.attn.to_add_out(text.contiguous())

    @staticmethod
    def _same_tensor(actual, expected):
        return (
            actual.shape == expected.shape
            and actual.stride() == expected.stride()
            and actual.data_ptr() == expected.data_ptr()
        )

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        encoder_hidden_states_mask=None,
        attention_mask=None,
        image_rotary_emb=None,
        *,
        overlap=True,
        overlap_q=True,
    ):
        if (
            self.closed
            or attn is not self.attn
            or not torch.is_inference_mode_enabled()
        ):
            raise RuntimeError("prepared Qwen AGKV requires its live inference slot")
        if torch.cuda.current_stream(hidden_states.device) != self.lane:
            raise RuntimeError("prepared Qwen AGKV requires its construction stream")
        if encoder_hidden_states_mask is not None or attention_mask is not None:
            raise ValueError("prepared Qwen AGKV requires compact unmasked text")
        if not self._same_tensor(
            hidden_states, self.hidden_states
        ) or not self._same_tensor(encoder_hidden_states, self.encoder_hidden_states):
            raise ValueError("prepare another Qwen AGKV slot for different storage")
        if image_rotary_emb is None or any(
            not self._same_tensor(actual, expected)
            for actual, expected in zip(image_rotary_emb, self.rotary_emb, strict=True)
        ):
            raise ValueError("prepare another Qwen AGKV slot for different RoPE")

        image_pending = self.image_projection.begin(
            hidden_states.flatten(0, 1),
            self.image_weight,
            self.image_bias,
            overlap=overlap,
            key_postprocess=self._postprocess_image_key,
        )
        text_pending = self.text_projection.begin(
            encoder_hidden_states.flatten(0, 1),
            self.text_weight,
            self.text_bias,
            overlap=overlap,
            key_postprocess=self._postprocess_text_key,
        )
        if not overlap_q:
            image_pending.wait()
            text_pending.wait()
        self.q_graph.replay()
        image_pending.wait()
        text_pending.wait()
        self.tail_graph.replay()
        return self.output

    def close(self):
        if self.closed:
            return
        self.image_projection.close()
        self.text_projection.close()
        self.closed = True


class PreparedQwenImagePackedAgkv:
    """Packed QKV candidate followed by one direct-slot CE all-gather."""

    def __init__(
        self,
        runtime,
        attn,
        hidden_states,
        encoder_hidden_states,
        image_rotary_emb,
        *,
        remote_streams=1,
        peer_order="xor",
        tag="qwen_packed_agkv",
    ):
        if not torch.is_inference_mode_enabled() or attn.training:
            raise ValueError("prepare Qwen AGKV in inference mode with eval weights")
        self.runtime = runtime
        self.attn = attn
        self.hidden_states = hidden_states
        self.encoder_hidden_states = encoder_hidden_states
        self.rotary_emb = image_rotary_emb
        self.lane = torch.cuda.current_stream(hidden_states.device)
        self.closed = False
        self.image_weight, self.image_bias = PreparedQwenImageShardedAgkv._pack(
            (attn.to_q, attn.to_k, attn.to_v)
        )
        self.text_weight, self.text_bias = PreparedQwenImageShardedAgkv._pack(
            (attn.add_q_proj, attn.add_k_proj, attn.add_v_proj)
        )
        setup_input = torch.cat([encoder_hidden_states, hidden_states], dim=1).flatten(
            0, 1
        )
        self.setup_input = setup_input
        kv_weight, kv_bias = PreparedQwenImageShardedAgkv._pack((attn.to_k, attn.to_v))
        width = attn.to_k.out_features
        self.projection = PersistentKVProjection(
            runtime,
            setup_input,
            kv_weight,
            heads=attn.heads,
            head_dim=width // attn.heads,
            bias=kv_bias,
            remote_streams=remote_streams,
            peer_order=peer_order,
            projection_backend="cublas",
            tag=tag,
        )
        local_shape = (1, setup_input.shape[0], attn.heads, width // attn.heads)
        self.qkv_out = (
            torch.empty(
                local_shape, device=hidden_states.device, dtype=hidden_states.dtype
            ),
            *(slot.view(local_shape) for slot in self.projection.local),
        )
        self.qkv_graph, local = _capture(self._qkv)
        self.query, self.local_key, self.local_value = local
        raw_key, value = self.projection.begin_direct(self.qkv_graph.replay).wait()
        self.attention = prepare_attention(
            self.query,
            raw_key,
            value,
            cp_degree=self.projection.world,
        )
        self.plan = self.attention.plan
        if self.plan.strategy == "sdpa":
            raise RuntimeError("Qwen packed AGKV requires the CuTe attention planner")

        def tail():
            output = _execute(
                self.plan,
                self.attention.forward,
                self.query,
                raw_key,
                value,
            )
            return self._finish(output)

        self.tail_graph, self.output = _capture(tail)
        self.metadata = {
            "attention_plan": asdict(self.plan),
            "partition": "image and text sequence-sharded",
            "projection_output": "packed QKV epilogue writes directly into symmetric K/V slots",
            "kv_staging_copy_bytes": 0,
            "local_key_postprocess": True,
            "kv_transport_q_overlap": False,
            "normalization": "paired QK RMSNorm+RoPE fused with joint QKV layout",
            "gemm_launches_qkv": 2,
        }

    def _qkv(self):
        return joint_qkv_norm_rope(
            F.linear(self.hidden_states, self.image_weight, self.image_bias),
            F.linear(self.encoder_hidden_states, self.text_weight, self.text_bias),
            self.attn,
            self.rotary_emb,
            out=self.qkv_out,
        )

    def _finish(self, output):
        text_tokens = self.encoder_hidden_states.shape[1]
        text = output[:, :text_tokens].flatten(2)
        image = output[:, text_tokens:].flatten(2)
        image = self.attn.to_out[0](image.contiguous())
        if len(self.attn.to_out) > 1:
            image = self.attn.to_out[1](image)
        return image, self.attn.to_add_out(text.contiguous())

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        encoder_hidden_states_mask=None,
        attention_mask=None,
        image_rotary_emb=None,
        **kwargs,
    ):
        del kwargs
        same = PreparedQwenImageShardedAgkv._same_tensor
        if (
            self.closed
            or attn is not self.attn
            or not torch.is_inference_mode_enabled()
        ):
            raise RuntimeError("prepared Qwen AGKV requires its live inference slot")
        if torch.cuda.current_stream(hidden_states.device) != self.lane:
            raise RuntimeError("prepared Qwen AGKV requires its construction stream")
        if encoder_hidden_states_mask is not None or attention_mask is not None:
            raise ValueError("prepared Qwen AGKV requires compact unmasked text")
        if not same(hidden_states, self.hidden_states) or not same(
            encoder_hidden_states, self.encoder_hidden_states
        ):
            raise ValueError("prepare another Qwen AGKV slot for different storage")
        if image_rotary_emb is None or any(
            not same(actual, expected)
            for actual, expected in zip(image_rotary_emb, self.rotary_emb, strict=True)
        ):
            raise ValueError("prepare another Qwen AGKV slot for different RoPE")
        pending = self.projection.begin_direct(self.qkv_graph.replay)
        pending.wait()
        self.tail_graph.replay()
        return self.output

    def close(self):
        if self.closed:
            return
        self.projection.close()
        self.closed = True


def prepare_qwen_image_sharded_agkv(
    runtime,
    attn,
    hidden_states,
    encoder_hidden_states,
    image_rotary_emb,
    *,
    schedule="auto",
    **kwargs,
):
    """Prepare a Qwen slot, timing packed/overlap once per runtime and shape.

    All ranks must call collectively, outside graph capture. Auto measures the
    complete block, including communication, and caches a rank-consistent choice.
    First use allocates both candidates' symmetric slots; those buffers remain
    owned by the runtime until it closes. Explicit schedules avoid this tuning.
    """
    classes = {
        "packed": PreparedQwenImagePackedAgkv,
        "overlap": PreparedQwenImageShardedAgkv,
    }
    if schedule not in ("auto", *classes):
        raise ValueError("Qwen AGKV schedule must be auto, overlap, or packed")

    def build(name):
        prepared = classes[name](
            runtime,
            attn,
            hidden_states,
            encoder_hidden_states,
            image_rotary_emb,
            **kwargs,
        )
        prepared.metadata["schedule"] = name
        return prepared

    if schedule != "auto":
        return build(schedule)
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("prepare Qwen AGKV before CUDA graph capture")
    pg = runtime._group.pg
    props = torch.cuda.get_device_properties(hidden_states.device)
    key = (
        tuple(hidden_states.shape),
        tuple(encoder_hidden_states.shape),
        hidden_states.dtype,
        attn.heads,
        attn.to_q.out_features,
        tuple(
            p.bias is not None
            for p in (
                attn.to_q,
                attn.to_k,
                attn.to_v,
                attn.add_q_proj,
                attn.add_k_proj,
                attn.add_v_proj,
            )
        ),
        tuple(
            n is not None
            for n in (
                attn.norm_q,
                attn.norm_k,
                attn.norm_added_q,
                attn.norm_added_k,
            )
        ),
        image_rotary_emb is not None,
        props.name,
        props.major,
        props.minor,
        props.multi_processor_count,
        dist.get_world_size(pg),
        kwargs.get("remote_streams", 1),
        kwargs.get("peer_order", "xor"),
    )
    cache = _schedule_cache.setdefault(runtime, {})
    cached = cache.get(key)
    choices = [None] * dist.get_world_size(pg)
    dist.all_gather_object(choices, cached[0] if cached else None, group=pg)
    if cached and all(choice == cached[0] for choice in choices):
        prepared = build(cached[0])
        prepared.metadata["schedule_tuning_ms"] = cached[1]
        prepared.metadata["schedule_cache_hit"] = True
        return prepared

    candidates = {}
    timings = {}
    try:
        for name in classes:
            prepared = candidates[name] = build(name)

            def call():
                prepared(
                    attn,
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    image_rotary_emb=image_rotary_emb,
                )

            for _ in range(3):
                call()
            samples = []
            for _ in range(5):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(3):
                    call()
                end.record()
                end.synchronize()
                elapsed = torch.tensor(
                    start.elapsed_time(end) / 3,
                    dtype=torch.float64,
                    device=hidden_states.device,
                )
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=pg)
                samples.append(float(elapsed))
            timings[name] = statistics.median(samples)
    except Exception:
        for candidate in candidates.values():
            candidate.close()
        raise
    selected = min(timings, key=timings.get)
    cache[key] = selected, timings
    for name, candidate in candidates.items():
        if name != selected:
            candidate.close()
    prepared = candidates[selected]
    prepared.metadata["schedule_tuning_ms"] = timings
    prepared.metadata["schedule_cache_hit"] = False
    return prepared


@dataclass(frozen=True)
class QwenImageAdapter:
    """Image and text are both evenly sequence-sharded; local order is text,image."""

    def signature(self):
        return ("qwen_image", "image_text_sharded", "text_first", "per_head_norm", 1)

    def validate(self, module, inputs, algorithm):
        x = validate_common(
            module,
            inputs,
            {
                "hidden_states",
                "encoder_hidden_states",
                "image_rotary_emb",
                "encoder_hidden_states_mask",
                "attention_mask",
            },
        )
        text = inputs.get("encoder_hidden_states")
        if (
            not isinstance(text, torch.Tensor)
            or text.ndim != 3
            or text.shape[0] != 1
            or text.shape[1] <= 0
            or text.shape[-1] != x.shape[-1]
            or not text.is_contiguous()
        ):
            raise UnsupportedFastCP(
                "Qwen requires a nonempty contiguous local text shard"
            )
        for name in ("add_q_proj", "add_k_proj", "add_v_proj"):
            layer = getattr(module, name, None)
            if (
                type(layer) is not torch.nn.Linear
                or layer.in_features != text.shape[-1]
                or layer.out_features != module.heads * 128
            ):
                raise UnsupportedFastCP(
                    "Qwen requires matching replicated text QKV projections"
                )
        per_head_norms(module, ("norm_q", "norm_k", "norm_added_q", "norm_added_k"))
        if type(module.to_add_out) is not torch.nn.Linear:
            raise UnsupportedFastCP("Qwen requires a replicated text output projection")
        rope = inputs.get("image_rotary_emb")
        if rope is None or len(rope) != 2 or any(t is None for t in rope):
            raise UnsupportedFastCP("Qwen prepared region requires image/text RoPE")
        _rope(rope[0], x.shape[1])
        _rope(rope[1], text.shape[1])

    def build(self, runtime, module, inputs, algorithm, schedule, tag):
        if algorithm == "agkv":
            p = prepare_qwen_image_sharded_agkv(
                runtime.backend,
                module,
                inputs["hidden_states"],
                inputs["encoder_hidden_states"],
                inputs["image_rotary_emb"],
                schedule=schedule,
                tag=tag,
                remote_streams=runtime.profile.remote_streams,
                peer_order=runtime.profile.peer_order,
            )
            return ProcessorExecution(module, p, p.metadata)
        from .....models.qwen_image.attention import QwenImageCpAttnProcessor

        p = QwenImageCpAttnProcessor(
            runtime.parallel(tag),
            mode="ulysses",
            planned_attention=True,
            sharded_text=True,
        )
        p.enable()
        return ProcessorExecution(
            module,
            p,
            {"schedule": "serial", "partition": "image and text sequence-sharded"},
            planner=p.attention,
        )
