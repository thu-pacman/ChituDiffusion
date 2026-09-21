"""One persistent KV GEMM with contiguous CE all-gather consumers."""

from dataclasses import dataclass

import torch
import torch.distributed as dist
import triton
import triton.language as tl


@dataclass(frozen=True)
class KVPlan:
    rows: int
    inner: int
    heads: int
    head_dim: int
    world: int
    chunk_tokens: int

    def __post_init__(self):
        if min(self.rows, self.inner, self.heads, self.head_dim) <= 0:
            raise ValueError("KV dimensions must be positive")
        if self.world not in (2, 4, 8) or self.width % 128:
            raise ValueError("KV requires CP2/4/8 and HD divisible by128")
        if self.chunk_tokens <= 0 or self.chunk_tokens % 128:
            raise ValueError("KV chunk must be a positive multiple of128")

    @property
    def width(self):
        return self.heads * self.head_dim

    @property
    def chunks(self):
        return triton.cdiv(self.rows, self.chunk_tokens)

    def tile_destination(self, tile, rank):
        mt, nt = divmod(tile, 2 * self.width // 128)
        projection, col_tile = divmod(nt, self.width // 128)
        return projection, rank * self.rows + mt * 128, col_tile * 128

    def expected_count(self, chunk, epoch):
        if not 0 <= chunk < self.chunks or epoch <= 0:
            raise ValueError("invalid KV chunk/epoch")
        count = min(self.chunk_tokens, self.rows - chunk * self.chunk_tokens)
        return epoch * triton.cdiv(count, 128) * (self.width // 128)

    def peer_groups(self, rank, remote_streams, peer_order):
        if not 0 <= rank < self.world or not 1 <= remote_streams < self.world:
            raise ValueError("invalid KV rank or CE stream count")
        if peer_order not in ("rotate", "xor"):
            raise ValueError("KV peer order must be rotate or xor")
        peers = tuple(
            rank ^ step if peer_order == "xor" else (rank + step) % self.world
            for step in range(1, self.world)
        )
        return tuple(peers[index::remote_streams] for index in range(remote_streams))


@triton.jit
def _persistent_kv_gemm(
    X,
    W,
    Bias,
    Key,
    Value,
    Counters,
    M: tl.constexpr,
    K: tl.constexpr,
    WIDTH: tl.constexpr,
    RANK: tl.constexpr,
    GRID: tl.constexpr,
    CHUNK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    PUBLISH: tl.constexpr,
    BM: tl.constexpr = 128,
    BN: tl.constexpr = 128,
    BK: tl.constexpr = 32,
):
    for tile in range(tl.program_id(0), tl.cdiv(M, BM) * (2 * WIDTH // BN), GRID):
        mt, nt = tile // (2 * WIDTH // BN), tile % (2 * WIDTH // BN)
        projection = nt // (WIDTH // BN)
        rm = mt * BM + tl.arange(0, BM)
        rn = nt * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.full((BM, BN), 0, tl.float32)
        for block in range(tl.cdiv(K, BK)):
            kk = block * BK + rk
            a = tl.load(
                X + rm[:, None] * K + kk[None, :],
                (rm[:, None] < M) & (kk[None, :] < K),
                0,
            )
            b = tl.load(W + rn[None, :] * K + kk[:, None], kk[:, None] < K, 0)
            acc += tl.dot(a, b)
        if HAS_BIAS:
            acc += tl.load(Bias + rn)[None, :]
        output = Key if projection == 0 else Value
        tl.store(
            output + (RANK * M + rm[:, None]) * WIDTH + rn[None, :] % WIDTH,
            acc.to(Key.dtype.element_ty),
            rm[:, None] < M,
        )
        if PUBLISH:
            tl.debug_barrier()
            tl.atomic_add(
                Counters + (mt * BM // CHUNK) * 2 + projection,
                1,
                sem="acq_rel",
                scope="sys",
            )


class KVProjectionHandle:
    """One pending epoch; wait on the construction stream before consuming KV."""

    def __init__(self, pipeline):
        self.pipeline, self.epoch = pipeline, pipeline.epoch
        self.done = False

    def wait(self):
        p = self.pipeline
        p._check_call()
        if self.epoch != p.epoch:
            raise RuntimeError("KV handle output has been reused by a later epoch")
        if not self.done:
            for event in p.sent:
                p.lane.wait_event(event)
            p._barrier()
            p._pending = None
            self.done = True
        return p.output


class PersistentKVProjection:
    """Inference-only prepared KV linear plus CE gather, with reusable output.

    Both producers write their result into the local source slot of the final
    symmetric K/V buffers. ``persistent`` publishes 128x128 tiles while its
    GEMM is running. ``cublas`` uses the tuned PyTorch GEMMs and starts CE after
    both local projections finish; CE can then overlap the following Q work.
    """

    def __init__(
        self,
        runtime,
        x,
        weight,
        *,
        heads,
        head_dim,
        bias=None,
        chunk_tokens=2048,
        tag="persistent_kv",
        remote_streams=None,
        peer_order="rotate",
        projection_backend="persistent",
    ):
        self.runtime, self.pg = runtime, runtime._group.pg
        self.rank, self.world = dist.get_rank(self.pg), dist.get_world_size(self.pg)
        self.plan = KVPlan(
            x.shape[0], x.shape[1], heads, head_dim, self.world, chunk_tokens
        )
        self.device, self.dtype, self.has_bias = x.device, x.dtype, bias is not None
        if projection_backend not in ("persistent", "cublas"):
            raise ValueError("projection_backend must be persistent or cublas")
        self.projection_backend = projection_backend
        self._validate(x, weight, bias)
        props = torch.cuda.get_device_properties(self.device)
        if (props.major, props.minor) != (9, 0):
            raise ValueError("KV experiment requires SM90")
        self.grid = props.multi_processor_count
        self.remote_streams = (
            self.world - 1 if remote_streams is None else remote_streams
        )
        self.peer_order = peer_order
        self.peer_groups = self.plan.peer_groups(
            self.rank, self.remote_streams, peer_order
        )
        signatures = [None] * self.world
        signature = (
            self.plan,
            self.dtype,
            self.has_bias,
            self.grid,
            tag,
            self.remote_streams,
            peer_order,
            projection_backend,
        )
        dist.all_gather_object(signatures, signature, group=self.pg)
        if any(item != signature for item in signatures):
            raise ValueError("KV projection plans differ across ranks")
        if not hasattr(torch.ops.fast_ulysses, "kv_projection_prepare"):
            raise RuntimeError("rebuild Fast CP with kv_projection.cu")
        if self.remote_streams < self.world - 1 and not hasattr(
            torch.ops.fast_ulysses, "kv_projection_send_ce_group"
        ):
            raise RuntimeError("rebuild Fast CP with grouped KV CE support")
        p = self.plan
        self.key, self.value, self.key_peers, self.value_peers = (
            torch.ops.fast_ulysses.kv_projection_prepare(
                runtime._group._group, x, p.rows, p.width, tag
            )
        )
        self.local = tuple(
            v[self.rank * p.rows : (self.rank + 1) * p.rows]
            for v in (self.key, self.value)
        )
        self.output = tuple(
            v.view(1, self.world * p.rows, heads, head_dim)
            for v in (self.key, self.value)
        )
        self.counters = torch.zeros(
            (p.chunks, 2), device=self.device, dtype=torch.int32
        )
        self.sync, self.sync_peers = torch.ops.fast_ulysses.projection_sync_prepare(
            runtime._group._group, x, tag + "_sync"
        )
        self.lane = torch.cuda.current_stream(self.device)
        self.senders = [
            torch.cuda.Stream(device=self.device) for _ in range(self.remote_streams)
        ]
        self.start, self.produced = torch.cuda.Event(), torch.cuda.Event()
        self.sent = [torch.cuda.Event() for _ in self.senders]
        self.epoch, self.closed, self.kernel = 0, False, None
        self._pending = None
        self.compute_only(x, weight, bias)

    def _validate(self, x, weight, bias):
        p = self.plan
        if x.shape != (p.rows, p.inner) or weight.shape != (2 * p.width, p.inner):
            raise ValueError("invalid KV input/weight shape")
        if x.dtype not in (torch.float16, torch.bfloat16) or x.dtype != self.dtype:
            raise ValueError("invalid KV dtype")
        if (bias is not None) != self.has_bias or (
            bias is not None and bias.shape != (2 * p.width,)
        ):
            raise ValueError("invalid KV bias")
        for tensor in (x, weight, *((bias,) if bias is not None else ())):
            if (
                tensor.device != self.device
                or tensor.dtype != self.dtype
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    "KV tensors must share CUDA device/dtype and be contiguous"
                )
            if tensor.requires_grad and torch.is_grad_enabled():
                raise ValueError("KV projection supports inference only")

    def _barrier(self):
        torch.ops.fast_ulysses.projection_sync(self.sync, self.sync_peers, self.rank)

    def _gemm(self, x, weight, bias, publish):
        p = self.plan
        if self.projection_backend == "cublas":
            for projection, output in enumerate(self.local):
                projection_weight = weight[
                    projection * p.width : (projection + 1) * p.width
                ]
                if bias is None:
                    torch.mm(x, projection_weight.t(), out=output)
                else:
                    projection_bias = bias[
                        projection * p.width : (projection + 1) * p.width
                    ]
                    torch.addmm(projection_bias, x, projection_weight.t(), out=output)
            self.kernel = None
            return
        self.kernel = _persistent_kv_gemm[(self.grid,)](
            x,
            weight,
            bias if bias is not None else x,
            self.key,
            self.value,
            self.counters,
            p.rows,
            p.inner,
            p.width,
            self.rank,
            self.grid,
            p.chunk_tokens,
            self.has_bias,
            publish,
            num_warps=8,
            num_stages=3,
        )

    def compute_only(self, x, weight, bias=None):
        if self._pending is not None:
            raise RuntimeError("wait for pending KV before reusing producer outputs")
        self._gemm(x, weight, bias, False)
        return self.local

    def _send(self, overlap, *, join=True):
        p = self.plan
        self.produced.record(self.lane)
        tile_overlap = overlap and self.projection_backend == "persistent"
        for index, stream in enumerate(self.senders):
            peers = self.peer_groups[index]
            stream.wait_event(self.start if tile_overlap else self.produced)
            with torch.cuda.stream(stream):
                grouped = len(peers) > 1
                sender = (
                    torch.ops.fast_ulysses.kv_projection_send_ce_group
                    if grouped
                    else torch.ops.fast_ulysses.kv_projection_send_ce
                )
                sender(
                    self.key,
                    self.value,
                    self.counters,
                    [self.key_peers[peer] for peer in peers]
                    if grouped
                    else self.key_peers[peers[0]],
                    [self.value_peers[peer] for peer in peers]
                    if grouped
                    else self.value_peers[peers[0]],
                    self.rank,
                    self.world,
                    self.epoch,
                    p.chunk_tokens,
                    tile_overlap,
                )
                self.sent[index].record()
            if join:
                self.lane.wait_event(self.sent[index])

    def transport_only(self):
        self._check_call()
        if self._pending is not None:
            raise RuntimeError("wait for pending KV before starting another send")
        self._barrier()
        self.start.record(self.lane)
        # Serial DMA does not read counters or advance their epoch.
        saved_epoch = self.epoch
        self.epoch = max(1, self.epoch)
        self._send(False)
        self.epoch = saved_epoch
        self._barrier()
        return self.output

    def _check_call(self):
        if self.closed or torch.cuda.current_stream(self.device) != self.lane:
            raise RuntimeError("KV pipeline requires its live construction stream")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("KV host epochs do not support Graph capture")

    def begin(self, x, weight, bias=None, *, overlap=True, key_postprocess=None):
        """Submit one GEMM and CE sends; defer inbound completion until wait()."""
        self._check_call()
        if self._pending is not None:
            raise RuntimeError("wait for pending KV before beginning another epoch")
        self._validate(x, weight, bias)
        p = self.plan
        if self.epoch + 1 >= (2**31 - 1) // (
            (p.chunk_tokens // 128) * (p.width // 128)
        ):
            raise RuntimeError("KV completion epoch exhausted; recreate pipeline")
        self.epoch += 1
        self._barrier()
        self.start.record(self.lane)
        self._gemm(x, weight, bias, True)
        if key_postprocess is not None:
            if self.projection_backend != "cublas":
                raise ValueError(
                    "key_postprocess requires the completion-gated cublas producer"
                )
            key_postprocess(self.local[0])
        handle = KVProjectionHandle(self)
        self._pending = handle
        self._send(overlap, join=False)
        return handle

    def begin_precomputed(self, key, value):
        """Copy completed local K/V into their final slots and start CE gather."""
        self._check_call()
        if self._pending is not None:
            raise RuntimeError("wait for pending KV before beginning another epoch")
        p = self.plan
        if self.epoch + 1 >= (2**31 - 1) // (
            (p.chunk_tokens // 128) * (p.width // 128)
        ):
            raise RuntimeError("KV completion epoch exhausted; recreate pipeline")
        expected = (p.rows, p.heads, p.head_dim)
        prepared = []
        for name, tensor in (("key", key), ("value", value)):
            if tensor.ndim == 4 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            if (
                tensor.shape != expected
                or tensor.device != self.device
                or tensor.dtype != self.dtype
            ):
                raise ValueError(f"precomputed {name} must have shape {expected}")
            prepared.append(tensor)
        self.epoch += 1
        self._barrier()
        self.start.record(self.lane)
        self.local[0].copy_(prepared[0].reshape(p.rows, p.width))
        self.local[1].copy_(prepared[1].reshape(p.rows, p.width))
        handle = KVProjectionHandle(self)
        self._pending = handle
        self._send(False, join=False)
        return handle

    def begin_direct(self, producer):
        """Produce final K/V in ``local`` on the construction stream, then gather.

        The callback must enqueue all writes, including epilogues, on that stream.
        A captured producer graph may be replayed here. Only the rank-local source
        slots may be written. The previous wait() joins outgoing reads, and stream
        order protects the previous local consumer. A barrier AFTER production
        joins all ranks' previous consumers before any new remote write. CE waits
        for producer completion; tile overlap is disabled.
        Do not write the slots before calling this method or while a send is pending.
        """
        self._check_call()
        if self._pending is not None:
            raise RuntimeError("wait for pending KV before beginning another epoch")
        p = self.plan
        if self.epoch + 1 >= (2**31 - 1) // (
            (p.chunk_tokens // 128) * (p.width // 128)
        ):
            raise RuntimeError("KV completion epoch exhausted; recreate pipeline")
        self.epoch += 1
        producer()
        self._barrier()
        self.start.record(self.lane)
        handle = KVProjectionHandle(self)
        self._pending = handle
        self._send(False, join=False)
        return handle

    def __call__(self, x, weight, bias=None, *, overlap=True, key_postprocess=None):
        return self.begin(
            x,
            weight,
            bias,
            overlap=overlap,
            key_postprocess=key_postprocess,
        ).wait()

    def close(self):
        if self.closed:
            return
        if self._pending is not None:
            self._pending.wait()
        self.lane.synchronize()
        for stream in self.senders:
            stream.synchronize()
        self.closed = True
