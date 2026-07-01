---
name: Async QKV-Comm Overlap
overview: Add compute/communication overlap to CP attention by interleaving Q/K/V projection with the AGKV all-gather and Ulysses all-to-all, driven by the model-agnostic core via projection callbacks on a dedicated CUDA comm stream. Behind a flag, bit-equivalent to today's serial path, with the real speedup landing on NVLink.
todos:
  - id: scaffold
    content: Add CHITU_CP_OVERLAP_QKV flag + lazy comm stream + attention_fused() dispatch with serial fallback in image_cp_attention.py (no behavior change yet)
    status: pending
  - id: agkv
    content: "Implement _agkv_fused: proj K -> K all-gather on comm stream, overlap V-proj/V-gather and Q-proj, event-sync, record_stream, then attention"
    status: pending
  - id: ulysses
    content: "Implement _ulysses_fused: overlap a2a of Q/K/V with projections, and overlap the output img a2a with txt all-gather-heads"
    status: pending
  - id: processor
    content: Build produce_q/k/v closures in z_image_attention.py and route to attention_fused when overlap on + mode in {agkv,ulysses} + mask fast-path; keep serial path otherwise
    status: pending
  - id: warmup
    content: Add a comm-stream NCCL warmup at enable to avoid first-use init stalls on the side stream
    status: pending
  - id: validate
    content: "Validate: bit-equivalence vs serial (g1 PSNR>60dB), no PCIe regression (cp4 AGKV / cp2,cp4 Ulysses g4 50-step), overlap visible in chrome trace"
    status: pending
  - id: later
    content: (Optional later) extend overlap to fp8 K/V, KV-cache fresh steps, ring/unified; reuse for Qwen-Image
    status: pending
isProject: false
---

# Async QKV-Comm Overlap for CP Attention (AGKV + Ulysses)

## Goal

Overlap the (cheap) Q/K/V projections with the (expensive) collective comm so that, per attention call:

- AGKV: `proj K -> launch K all-gather; proj V (overlaps K-comm) -> launch V all-gather; proj Q (overlaps V-comm); wait K,V; attention`.
- Ulysses: `proj K -> a2a K; proj V -> a2a V; proj Q -> a2a Q (each proj overlaps the previous a2a); wait; attention`.

On PCIe (no P2P) the worklog measured the overlap ceiling at ~4% (compute << comm), so this is a near-neutral change here; the payoff is on NVLink where comm is cheap and proj is a real share of attention time.

## Current flow (what changes)

- Processor [chitu_diffusion/modules/attention/z_image_attention.py](chitu_diffusion/modules/attention/z_image_attention.py) does proj + `norm_q/k` + complex RoPE, then passes finished, pre-split tensors to the core:

```167:169:chitu_diffusion/modules/attention/z_image_attention.py
            txt_out, img_out = self._cp.attention(
                query[:, img_len:], key[:, img_len:], value[:, img_len:],
                query[:, :img_len], key[:, :img_len], value[:, :img_len],
            )
```

- Core [chitu_diffusion/modules/attention/image_cp_attention.py](chitu_diffusion/modules/attention/image_cp_attention.py) runs the collectives synchronously on the current stream: AGKV gathers K then V back-to-back (`_attn_agkv`, lines 480-496); Ulysses does three serial `_all_to_all` (lines 498-527). `comm_group.all_gather/all_to_all` are plain blocking calls (no handles), so overlap requires a separate comm stream + CUDA events.
- Profiling (`time()`/`_comm_time()`) calls `synchronize()`, which would defeat overlap -> the overlapped path must be the no-profile path (profiling keeps the serial path, exactly like ring does today).
- Only Z-Image uses the core today (Qwen-Image does not yet), so wiring is contained; the API stays model-agnostic.

## Design: core-driven callbacks + comm stream

Add a new core entry point that owns the schedule; the model supplies projection closures:

```python
# image_cp_attention.py (new)
def attention_fused(self, *, img_len, produce_q, produce_k, produce_v):
    # produce_x() -> packed local [B, S_local, H, D] over [image_chunk, text],
    # already normed / RoPE'd / contiguous (same tensor the processor builds today).
    if self.profile or not self._overlap or self.mode not in ("agkv", "ulysses"):
        q, k, v = produce_q(), produce_k(), produce_v()            # serial fallback
        return self.attention(q[:, img_len:], k[:, img_len:], v[:, img_len:],
                              q[:, :img_len], k[:, :img_len], v[:, :img_len])
    return self._agkv_fused(img_len, produce_q, produce_k, produce_v) if self.mode == "agkv" \
        else self._ulysses_fused(img_len, produce_q, produce_k, produce_v)
```

Comm stream + event mechanics (illustrative, AGKV):

```python
cs = self._comm_stream
k = produce_k(); k_img, k_txt = k[:, :img_len], k[:, img_len:]
k_img.record_stream(cs); cs.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(cs):
    full_k = self.all_gather_seq(k_img, grp, kind="kv")    # NCCL on comm stream
ev_k = torch.cuda.Event(); ev_k.record(cs)
v = produce_v(); v_img, v_txt = v[:, :img_len], v[:, img_len:]   # overlaps K-comm
... same for full_v on cs, ev_v ...
q = produce_q()                                                  # overlaps V-comm
torch.cuda.current_stream().wait_event(ev_k); torch.cuda.current_stream().wait_event(ev_v)
for t in (full_k, full_v): t.record_stream(torch.cuda.current_stream())
# assemble [img, txt], run attn_backend on compute stream, slice outputs (same as _attn_agkv)
```

Gating: env `CHITU_CP_OVERLAP_QKV` (default off initially); a `self._comm_stream` created lazily. Modes other than agkv/ulysses, plus profiling, fp8, and KV-cache, take the serial fallback in phase 1.

## File-level changes

- [chitu_diffusion/modules/attention/image_cp_attention.py](chitu_diffusion/modules/attention/image_cp_attention.py):
  - `__init__`: read `CHITU_CP_OVERLAP_QKV`, add `self._comm_stream=None` (lazy-created on first CUDA use).
  - Add `attention_fused(...)` dispatch + `_agkv_fused(...)` + `_ulysses_fused(...)`. Ulysses also overlaps the two output collectives (`img_out` a2a vs `txt_out` all-gather-heads).
  - Reuse existing `all_gather_seq` / `_all_to_all` bodies; only the streaming/eventing is new. Keep `_attn_agkv`/`_attn_ulysses` untouched for the serial/profile path.
  - Preserve all profiling buckets and `stats()` keys unchanged (overlap path simply does not profile).
- [chitu_diffusion/modules/attention/z_image_attention.py](chitu_diffusion/modules/attention/z_image_attention.py):
  - In `__call__`, build three closures `produce_q/k/v` (each: `to_x` -> `unflatten` -> `norm_x` if applicable -> RoPE on q,k only -> cast/contiguous), reusing `_apply_rotary_emb`.
  - When CP enabled, mask fast-path, overlap flag on, and mode in {agkv,ulysses}: call `self._cp.attention_fused(img_len=..., produce_q=..., produce_k=..., produce_v=...)`; else keep the current `self._cp.attention(...)` path verbatim.
  - The `qkv` profiling bucket: in serial/profile mode keep wrapping with `self._cp.time(_qkv_proj, "qkv")` as today.

## Phasing

1. Scaffolding: flag + lazy comm stream + `attention_fused` with serial fallback only (no behavior change). Verify Z-Image still bit-identical.
2. `_agkv_fused` overlap + processor closures + wiring. Validate.
3. `_ulysses_fused` overlap (input Q/K/V a2a + output a2a/all-gather-heads). Validate.
4. (Later, optional) extend overlap to fp8 K/V (quantize on compute, gather bytes+scales on comm stream), KV-cache fresh steps, and ring/unified; reuse for Qwen-Image.

## Validation (assumption: bit-equivalence-first, environment-agnostic)

- Correctness: overlap path is the same math reordered -> assert near bit-identical vs serial CP (`CHITU_CP_OVERLAP_QKV=0` vs `=1`) at g1: expect PSNR > 60 dB (ideally inf), and unchanged ~47 dB vs single-GPU.
- No PCIe regression: cp4 AGKV + cp2/cp4 Ulysses g4 50-step denoise within noise of current numbers.
- Overlap visible: with `CHITU_TORCH_TRACE`, confirm the all-gather/a2a NCCL kernel runs concurrently with the proj GEMM on the chrome trace.
- NVLink (if available): same configs should show attention-time reduction; otherwise provide an analytical projection from the `qkv` vs `kv`/`a2a` profile buckets.

## Risks and mitigations

- Multi-stream tensor lifetime (allocator may free a tensor still in use by the other stream) -> disciplined `record_stream` on every cross-stream tensor (proj outputs read by comm; gathered outputs read by compute); covered by the bit-equivalence test.
- NCCL on a side stream first-use init -> one warmup collective on the comm stream at enable time (mirrors the ring-graph warmup at lines 796-805).
- Profiling defeats overlap -> overlap strictly gated to `profile=False`; buckets/keys untouched.
- fp8 / KV-cache / ring / ring_graph / unified -> explicit serial fallback in phase 1 (correct, just not overlapped).
- CUDA-graph (`ring_graph`) unaffected (different mode).
