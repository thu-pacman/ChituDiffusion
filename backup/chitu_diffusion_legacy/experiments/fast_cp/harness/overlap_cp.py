"""Async QKV-comm overlap implemented as a *subclass* of the real CP core.

Kept entirely inside the harness directory: we do NOT modify
``image_cp_attention.py`` while debugging. Once the schedule here is proven
(bit-equivalent + visible overlap + speedup), it gets ported into the core's
``attention_fused()`` per ``async_comm_h20_nvlink_plan.md`` step 5.

``OverlapCPAttention`` adds:
  * a lazily-created CUDA comm stream;
  * ``attention_fused(img_len, produce_q, produce_k, produce_v)`` that interleaves
    the Q/K/V projection closures with the AGKV all-gather / Ulysses all-to-all
    on the comm stream, synchronized by CUDA events.

The serial / profile / unsupported paths fall back to ``cp.attention(...)`` via
the eager closures, so the subclass is bit-equivalent to the core by construction.
"""
from __future__ import annotations

import os

import torch

from chitu_diffusion.modules.attention.image_cp_attention import (
    ImageContextParallelAttention,
)


def _gp(msg):
    """Unbuffered, rank-aware progress print for diagnosing graph-capture hangs."""
    import os
    import sys
    if os.environ.get("CHITU_GRAPH_DEBUG", "0") == "1":
        r = os.environ.get("RANK", "?")
        print(f"[graph-dbg rank{r}] {msg}", flush=True)
        sys.stderr.flush()


class OverlapCPAttention(ImageContextParallelAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._comm_stream = None
        self._graph_cache = {}
        self._compile_tail = os.environ.get("CHITU_CP_COMPILE_TAIL", "0") == "1"
        self._compiled_tail = None
        self._compile_warned = False
        split = os.environ.get("CHITU_ULYSSES_SPLIT", "none").strip().lower()
        self._ulysses_split = {
            "0": "none",
            "false": "none",
            "off": "none",
            "input": "txt_q",
            "q": "txt_q",
            "output": "txt_out",
            "out": "txt_out",
            "both": "txt_both",
        }.get(split, split)
        if self._ulysses_split not in ("none", "txt_q", "txt_out", "txt_both"):
            raise ValueError(
                "CHITU_ULYSSES_SPLIT must be one of none/txt_q/txt_out/txt_both, "
                f"got {split!r}"
            )

    def _agkv_tail(self, img_len, q, full_k, full_v, k_txt, v_txt):
        local_img_len = q[:, :img_len].shape[1]
        qq = torch.cat([q[:, :img_len], q[:, img_len:]], dim=1).contiguous()
        kk = torch.cat([full_k, k_txt], dim=1).contiguous()
        vv = torch.cat([full_v, v_txt], dim=1).contiguous()
        out = self.attn_backend(qq, kk, vv, causal=False)[0]
        return out[:, local_img_len:], out[:, :local_img_len]

    def _get_agkv_tail(self):
        if not self._compile_tail:
            return self._agkv_tail
        if self._compiled_tail is None:
            try:
                self._compiled_tail = torch.compile(
                    self._agkv_tail,
                    mode=os.environ.get("CHITU_CP_COMPILE_MODE", "default"),
                    fullgraph=False,
                    dynamic=False,
                )
            except Exception as exc:
                if not self._compile_warned:
                    print(f"[overlap-cp] torch.compile tail disabled: {exc!r}", flush=True)
                    self._compile_warned = True
                self._compile_tail = False
                return self._agkv_tail
        return self._compiled_tail

    # ------------------------------------------------------------------ #
    def _get_comm_stream(self):
        if self._comm_stream is None:
            self._comm_stream = torch.cuda.Stream()
        return self._comm_stream

    def _serial_fused(self, img_len, produce_q, produce_k, produce_v):
        grp = self._group
        if hasattr(grp, "reset_a2a_sequence"):
            grp.reset_a2a_sequence()
        q, k, v = produce_q(), produce_k(), produce_v()
        return self.attention(
            q[:, img_len:], k[:, img_len:], v[:, img_len:],
            q[:, :img_len], k[:, :img_len], v[:, :img_len],
        )

    # ------------------------------------------------------------------ #
    def attention_fused(self, *, img_len, produce_q, produce_k, produce_v):
        grp = self._group
        cp = grp.group_size
        # Fall back to the exact serial path for everything not in scope.
        if (
            self.profile
            or self.mode not in ("agkv", "ulysses")
            or self._fp8_kv
            or self._kvcache_enabled
            or cp <= 1
        ):
            return self._serial_fused(img_len, produce_q, produce_k, produce_v)
        if self.mode == "agkv":
            return self._agkv_fused(img_len, produce_q, produce_k, produce_v)
        # ulysses needs heads divisible by cp; otherwise fall back (the core
        # itself would fall back to agkv, but for the overlap MVP we just stay
        # serial so the schedule under test is unambiguous).
        probe = produce_q()
        if probe.shape[2] % cp != 0:
            return self._serial_fused(img_len, produce_q, produce_k, produce_v)
        return self._ulysses_fused(img_len, produce_q, produce_k, produce_v, q_pre=probe)

    # ------------------------------------------------------------------ #
    def _agkv_fused(self, img_len, produce_q, produce_k, produce_v):
        grp = self._group
        cur = torch.cuda.current_stream()
        cs = self._get_comm_stream()

        # K: project on compute, all-gather image K on comm stream.
        k = produce_k()
        k_img, k_txt = k[:, :img_len], k[:, img_len:]
        cs.wait_stream(cur)
        k_img.record_stream(cs)
        with torch.cuda.stream(cs):
            full_k = self.all_gather_seq(k_img, grp, kind="kv")
        ev_k = torch.cuda.Event()
        ev_k.record(cs)

        # V projection overlaps the K all-gather.
        v = produce_v()
        v_img, v_txt = v[:, :img_len], v[:, img_len:]
        cs.wait_stream(cur)
        v_img.record_stream(cs)
        with torch.cuda.stream(cs):
            full_v = self.all_gather_seq(v_img, grp, kind="kv")
        ev_v = torch.cuda.Event()
        ev_v.record(cs)

        # Q projection overlaps the V all-gather.
        q = produce_q()
        local_img_len = q[:, :img_len].shape[1]

        cur.wait_event(ev_k)
        cur.wait_event(ev_v)
        full_k.record_stream(cur)
        full_v.record_stream(cur)

        return self._get_agkv_tail()(img_len, q, full_k, full_v, k_txt, v_txt)

    # ------------------------------------------------------------------ #
    def _ulysses_fused(self, img_len, produce_q, produce_k, produce_v, q_pre=None):
        grp = self._group
        if hasattr(grp, "reset_a2a_sequence"):
            grp.reset_a2a_sequence()
        cp = grp.group_size
        r = grp.rank_in_group
        cur = torch.cuda.current_stream()
        cs = self._get_comm_stream()

        def a2a_on_comm(t):
            cs.wait_stream(cur)
            t.record_stream(cs)
            with torch.cuda.stream(cs):
                out = grp.all_to_all(t, scatter_dim=2, gather_dim=1)
            ev = torch.cuda.Event()
            ev.record(cs)
            return out, ev

        # K
        k = produce_k()
        k_img, k_txt = k[:, :img_len], k[:, img_len:]
        a2a_k, ev_k = a2a_on_comm(k_img)

        # V overlaps K a2a
        v = produce_v()
        v_img, v_txt = v[:, :img_len], v[:, img_len:]
        a2a_v, ev_v = a2a_on_comm(v_img)

        # Q overlaps V a2a
        q = q_pre if q_pre is not None else produce_q()
        q_img, q_txt = q[:, :img_len], q[:, img_len:]
        a2a_q, ev_q = a2a_on_comm(q_img)

        return self._ulysses_finish_attention(
            a2a_k, ev_k, a2a_v, ev_v, a2a_q, ev_q, q_txt, k_txt, v_txt
        )

    def _ulysses_text_heads(self, q_txt, k_txt, v_txt):
        grp = self._group
        cp = grp.group_size
        r = grp.rank_in_group
        # Text is replicated before Ulysses; each rank keeps its local head
        # slice to match the image all-to-all head sharding.
        txt_q = torch.tensor_split(q_txt, cp, dim=2)[r].contiguous()
        txt_k = torch.tensor_split(k_txt, cp, dim=2)[r].contiguous()
        txt_v = torch.tensor_split(v_txt, cp, dim=2)[r].contiguous()
        return txt_q, txt_k, txt_v

    def _ulysses_finish_attention(self, a2a_k, ev_k, a2a_v, ev_v, a2a_q, ev_q, q_txt, k_txt, v_txt):
        grp = self._group
        cur = torch.cuda.current_stream()
        cs = self._get_comm_stream()

        def wait_one(ev, t):
            cur.wait_event(ev)
            t.record_stream(cur)

        def wait_kv():
            wait_one(ev_k, a2a_k)
            wait_one(ev_v, a2a_v)

        def wait_q():
            wait_one(ev_q, a2a_q)

        def image_out_a2a_on_comm(img_shard):
            cs.wait_stream(cur)
            img_shard.record_stream(cs)
            with torch.cuda.stream(cs):
                out = grp.all_to_all(img_shard, scatter_dim=1, gather_dim=2)
            ev = torch.cuda.Event()
            ev.record(cs)
            return out, ev

        txt_q, txt_k, txt_v = self._ulysses_text_heads(q_txt, k_txt, v_txt)

        if self._ulysses_split == "none":
            wait_kv()
            wait_q()
            full_img_len = a2a_q.shape[1]
            qq = torch.cat([a2a_q, txt_q], dim=1)
            kk = torch.cat([a2a_k, txt_k], dim=1)
            vv = torch.cat([a2a_v, txt_v], dim=1)
            out = self.attn_backend(qq, kk, vv, causal=False)[0]
            img_out = grp.all_to_all(out[:, :full_img_len], scatter_dim=1, gather_dim=2)
            txt_out = self._all_gather_heads(out[:, full_img_len:])
            return txt_out.contiguous(), img_out.contiguous()

        wait_kv()
        kk = torch.cat([a2a_k, txt_k], dim=1)
        vv = torch.cat([a2a_v, txt_v], dim=1)

        if self._ulysses_split == "txt_q":
            # Text query attention does not depend on image Q, so run it while
            # the image-Q all-to-all is still in flight.
            txt_out = self.attn_backend(txt_q, kk, vv, causal=False)[0]
            wait_q()
            img_shard = self.attn_backend(a2a_q, kk, vv, causal=False)[0]
            img_out = grp.all_to_all(img_shard, scatter_dim=1, gather_dim=2)
            txt_out = self._all_gather_heads(txt_out)
            return txt_out.contiguous(), img_out.contiguous()

        if self._ulysses_split == "txt_out":
            wait_q()
            img_shard = self.attn_backend(a2a_q, kk, vv, causal=False)[0]
            img_out, ev_img_out = image_out_a2a_on_comm(img_shard)
            # Text query attention overlaps the image-output all-to-all.
            txt_out = self.attn_backend(txt_q, kk, vv, causal=False)[0]
            wait_one(ev_img_out, img_out)
            txt_out = self._all_gather_heads(txt_out)
            return txt_out.contiguous(), img_out.contiguous()

        # Split text queries so part of text compute can cover image-Q input
        # a2a and the rest can cover image-output a2a.
        split_at = max(1, txt_q.shape[1] // 2)
        txt_q0 = txt_q[:, :split_at]
        txt_q1 = txt_q[:, split_at:]
        txt_out0 = self.attn_backend(txt_q0, kk, vv, causal=False)[0]
        wait_q()
        img_shard = self.attn_backend(a2a_q, kk, vv, causal=False)[0]
        img_out, ev_img_out = image_out_a2a_on_comm(img_shard)
        if txt_q1.shape[1] > 0:
            txt_out1 = self.attn_backend(txt_q1, kk, vv, causal=False)[0]
            txt_out = torch.cat([txt_out0, txt_out1], dim=1)
        else:
            txt_out = txt_out0
        wait_one(ev_img_out, img_out)
        txt_out = self._all_gather_heads(txt_out)
        return txt_out.contiguous(), img_out.contiguous()

    # ------------------------------------------------------------------ #
    # CUDA graph capture of the AGKV overlap schedule.
    #
    # Captures the WHOLE region proj(q/k/v) -> gather(k,v) on comm stream ->
    # attention, so the per-op CPU launch + event/stream overhead that makes the
    # eager overlap path lose on NVLink is removed by replay. The static input is
    # `hidden_states`; the harness copies fresh data into it before each replay.
    # Mirrors the warmup->capture->replay discipline of the core's
    # `_capture_ring_graph` (NCCL connections established on a side stream before
    # capture so they are not created illegally inside the graph).
    # ------------------------------------------------------------------ #
    def attention_fused_graphed(self, *, img_len, hidden_states, produce_packed, produce_one=None):
        """``produce_one(hs, which)`` builds one packed ``[B, S, H, D]`` tensor
        from ``hidden_states`` (proj + norm + rope).
        The whole proj->overlap-gather->attention region is captured once per
        shape and replayed. Returns ``(txt_out, img_out)`` like ``attention``.
        """
        if produce_one is None:
            produce_one = lambda hs, which: produce_packed(hs)[{"q": 0, "k": 1, "v": 2}[which]]
        key = (self.mode, self._ulysses_split, tuple(hidden_states.shape), hidden_states.dtype, img_len)
        entry = self._graph_cache.get(key)
        if entry is None:
            if self.mode == "agkv":
                entry = self._capture_agkv_graph(img_len, hidden_states, produce_one)
            elif self.mode == "ulysses":
                entry = self._capture_ulysses_graph(img_len, hidden_states, produce_one)
            else:
                raise ValueError(f"graphed fused path only supports agkv/ulysses, got {self.mode!r}")
            self._graph_cache[key] = entry
        entry["hs"].copy_(hidden_states)
        entry["graph"].replay()
        return entry["txt_out"].clone(), entry["img_out"].clone()

    def _agkv_overlap_from_hidden(self, img_len, hs, produce_one):
        """The eager AGKV overlap schedule, but taking a single hidden_states and
        producing q/k/v inside (so the projection GEMMs are inside the captured
        region). Same stream/event discipline as ``_agkv_fused``."""
        grp = self._group
        cur = torch.cuda.current_stream()
        cs = self._get_comm_stream()

        k = produce_one(hs, "k")
        k_img, k_txt = k[:, :img_len], k[:, img_len:]
        cs.wait_stream(cur)
        k_img.record_stream(cs)
        with torch.cuda.stream(cs):
            full_k = self.all_gather_seq(k_img, grp, kind="kv")
        ev_k = torch.cuda.Event()
        ev_k.record(cs)

        # V projection overlaps K all-gather.
        v = produce_one(hs, "v")
        v_img, v_txt = v[:, :img_len], v[:, img_len:]
        cs.wait_stream(cur)
        v_img.record_stream(cs)
        with torch.cuda.stream(cs):
            full_v = self.all_gather_seq(v_img, grp, kind="kv")
        ev_v = torch.cuda.Event()
        ev_v.record(cs)

        # Q projection overlaps V all-gather.
        q = produce_one(hs, "q")
        cur.wait_event(ev_k)
        cur.wait_event(ev_v)
        full_k.record_stream(cur)
        full_v.record_stream(cur)

        return self._get_agkv_tail()(img_len, q, full_k, full_v, k_txt, v_txt)

    def _capture_agkv_graph(self, img_len, hidden_states, produce_one):
        _gp("agkv capture: start")
        s_hs = hidden_states.clone()

        # Warm up on a side stream first: required before capture; also
        # establishes NCCL connections so they are not created during capture.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._agkv_overlap_from_hidden(img_len, s_hs, produce_one)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        _gp("agkv capture: warmup done")

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            s_txt_out, s_img_out = self._agkv_overlap_from_hidden(
                img_len, s_hs, produce_one
            )
        _gp("agkv capture: capture done")
        torch.cuda.synchronize()
        _gp("agkv capture: synced, returning")
        return {"graph": graph, "hs": s_hs, "txt_out": s_txt_out, "img_out": s_img_out}

    # ------------------------------------------------------------------ #
    # Ulysses graph capture: input-side async only.
    #
    # Captures K/V/Q projection staggered with K/V/Q image all-to-all on the
    # comm stream, then keeps output collectives serial exactly like the eager
    # phase-1 Ulysses path.
    # ------------------------------------------------------------------ #
    def _ulysses_from_hidden(self, img_len, hs, produce_one):
        grp = self._group
        if hasattr(grp, "reset_a2a_sequence"):
            grp.reset_a2a_sequence()
        cp = grp.group_size
        r = grp.rank_in_group
        cur = torch.cuda.current_stream()
        cs = self._get_comm_stream()

        def a2a_on_comm(t):
            cs.wait_stream(cur)
            t.record_stream(cs)
            with torch.cuda.stream(cs):
                out = grp.all_to_all(t, scatter_dim=2, gather_dim=1)
            ev = torch.cuda.Event()
            ev.record(cs)
            return out, ev

        k = produce_one(hs, "k")
        k_img, k_txt = k[:, :img_len], k[:, img_len:]
        a2a_k, ev_k = a2a_on_comm(k_img)

        # V projection overlaps K a2a.
        v = produce_one(hs, "v")
        v_img, v_txt = v[:, :img_len], v[:, img_len:]
        a2a_v, ev_v = a2a_on_comm(v_img)

        # Q projection overlaps V a2a.
        q = produce_one(hs, "q")
        q_img, q_txt = q[:, :img_len], q[:, img_len:]
        a2a_q, ev_q = a2a_on_comm(q_img)

        return self._ulysses_finish_attention(
            a2a_k, ev_k, a2a_v, ev_v, a2a_q, ev_q, q_txt, k_txt, v_txt
        )

    def _capture_ulysses_graph(self, img_len, hidden_states, produce_one):
        _gp("ulysses capture: start")
        s_hs = hidden_states.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._ulysses_from_hidden(img_len, s_hs, produce_one)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        _gp("ulysses capture: warmup done")

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            s_txt_out, s_img_out = self._ulysses_from_hidden(
                img_len, s_hs, produce_one
            )
        _gp("ulysses capture: capture done")
        torch.cuda.synchronize()
        _gp("ulysses capture: synced, returning")
        return {"graph": graph, "hs": s_hs, "txt_out": s_txt_out, "img_out": s_img_out}

    # ------------------------------------------------------------------ #
    # Serial (no-overlap) graphed baseline: same captured region, but q/k/v are
    # gathered back-to-back on the current stream (the eager `_attn_agkv` shape).
    # Lets us separate "graph removes CPU launch overhead" from "overlap hides
    # comm" -- compare serial-graph vs overlap-graph.
    # ------------------------------------------------------------------ #
    def attention_serial_graphed(self, *, img_len, hidden_states, produce_packed):
        key = ("serial", tuple(hidden_states.shape), hidden_states.dtype, img_len)
        entry = self._graph_cache.get(key)
        if entry is None:
            entry = self._capture_serial_graph(img_len, hidden_states, produce_packed)
            self._graph_cache[key] = entry
        entry["hs"].copy_(hidden_states)
        entry["graph"].replay()
        return entry["txt_out"].clone(), entry["img_out"].clone()

    def _serial_from_hidden(self, img_len, hs, produce_packed):
        grp = self._group
        if hasattr(grp, "reset_a2a_sequence"):
            grp.reset_a2a_sequence()
        q, k, v = produce_packed(hs)
        return self.attention(
            q[:, img_len:], k[:, img_len:], v[:, img_len:],
            q[:, :img_len], k[:, :img_len], v[:, :img_len],
        )

    def _capture_serial_graph(self, img_len, hidden_states, produce_packed):
        _gp("serial capture: start")
        s_hs = hidden_states.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._serial_from_hidden(img_len, s_hs, produce_packed)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        _gp("serial capture: warmup done")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            s_txt_out, s_img_out = self._serial_from_hidden(img_len, s_hs, produce_packed)
        _gp("serial capture: capture done")
        torch.cuda.synchronize()
        _gp("serial capture: synced, returning")
        return {"graph": graph, "hs": s_hs, "txt_out": s_txt_out, "img_out": s_img_out}
