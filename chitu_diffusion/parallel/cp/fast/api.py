"""Explicit preparation and ownership for inference-only Fast CP regions.

All ranks prepare the same regions in the same order. Forward never discovers
or rewrites Python code and never switches communication algorithms on failure.
"""

from __future__ import annotations

import hashlib
import socket
import threading
import weakref
from contextlib import contextmanager
from types import SimpleNamespace

import torch
import torch.distributed as dist

from .adapters.base import UnsupportedFastCP
from .profiles import H20_NVLINK

_owner = None


def _map(tree, fn):
    if isinstance(tree, torch.Tensor):
        return fn(tree)
    if isinstance(tree, dict):
        return {k: _map(v, fn) for k, v in tree.items()}
    if isinstance(tree, (tuple, list)):
        return type(tree)(_map(v, fn) for v in tree)
    if tree is None or isinstance(tree, (str, int, float, bool)):
        return tree
    raise TypeError(f"unsupported region input/output type: {type(tree).__name__}")


def _signature(tree):
    if isinstance(tree, torch.Tensor):
        return (
            "tensor",
            tuple(tree.shape),
            tuple(tree.stride()),
            str(tree.dtype),
            tree.device.type,
        )
    if isinstance(tree, dict):
        return tuple((k, _signature(v)) for k, v in sorted(tree.items()))
    if isinstance(tree, (tuple, list)):
        return (type(tree).__name__, tuple(_signature(v) for v in tree))
    if tree is None or isinstance(tree, (str, int, float, bool)):
        return tree
    raise TypeError(f"unsupported input type: {type(tree).__name__}")


def _copy_into(dst, src):
    if isinstance(dst, torch.Tensor):
        if dst.device != src.device:
            raise ValueError("input device changed; prepare another execution")
        dst.copy_(src)
    elif isinstance(dst, dict):
        for k in dst:
            _copy_into(dst[k], src[k])
    elif isinstance(dst, (tuple, list)):
        for a, b in zip(dst, src, strict=True):
            _copy_into(a, b)


def _module_signature(module):
    params = tuple(
        (n, tuple(p.shape), str(p.dtype)) for n, p in module.named_parameters()
    )
    semantics = tuple(
        (
            n,
            type(m).__module__,
            type(m).__qualname__,
            getattr(m, "eps", None),
            getattr(m, "p", None),
        )
        for n, m in module.named_modules()
    )
    return (
        params,
        semantics,
        getattr(module, "heads", None),
        getattr(module, "scale", None),
    )


class _SerialUlyssesTransport:
    """Borrow a backend, with no async protocol and no ownership of its heap."""

    name = "fast_ulysses"

    def __init__(self, backend, tag):
        self.backend, self.tag, self.index = backend, tag, 0

    def reset(self):
        self.index = 0

    def all_to_all(self, tensor, scatter_dim, gather_dim):
        tag = f"{self.tag}_u{self.index}"
        self.index += 1
        return self.backend.all_to_all_tagged(tensor, scatter_dim, gather_dim, tag=tag)

    def all_to_all_qkv(self, query, key, value):
        return tuple(self.all_to_all(t, 2, 1) for t in (query, key, value))

    def close(self):
        pass


class FastCPRuntime:
    def __init__(
        self,
        process_group=None,
        *,
        device=None,
        hardware_profile="h20_nvlink",
        backend=None,
        pool_bytes=2 << 30,
        max_workspaces=8,
    ):
        if hardware_profile != "h20_nvlink":
            raise ValueError("only the calibrated h20_nvlink profile is available")
        if pool_bytes <= 0 or max_workspaces <= 0:
            raise ValueError("pool_bytes and max_workspaces must be positive")
        self.process_group = process_group
        if dist.is_initialized() and process_group is None:
            self.process_group = dist.group.WORLD
        self.world = (
            dist.get_world_size(self.process_group) if dist.is_initialized() else 1
        )
        self.rank = dist.get_rank(self.process_group) if dist.is_initialized() else 0
        self.device = torch.device(
            device
            if device is not None
            else (
                f"cuda:{torch.cuda.current_device()}"
                if torch.cuda.is_available()
                else "cpu"
            )
        )
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.profile = H20_NVLINK
        self._backend, self._owns_backend = backend, False
        if backend is not None and (
            backend._process_group is not self.process_group
            or backend._world_size != self.world
        ):
            raise ValueError("borrowed backend must use this exact process group")
        self.pool_bytes, self.max_workspaces = pool_bytes, max_workspaces
        self._workspaces = {}
        self._blocks = []
        self._lock = threading.Lock()
        self.closed = False
        self.lane = (
            torch.cuda.current_stream(self.device)
            if self.device.type == "cuda"
            else None
        )
        # PyTorch exposes the raw stream getter to generated CUDA launchers.
        # Keep the public stream API fallback for builds without that entry point.
        self._raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)
        self._lane_handle = self.lane.cuda_stream if self.lane is not None else None
        if backend is not None:
            self._attach_backend()
        self._device_reason = self._probe_device()

    def _attach_backend(self):
        state = getattr(self.backend, "_fast_cp_wrapper_state", None)
        if state is None:
            state = SimpleNamespace(
                lock=self._lock,
                workspaces=self._workspaces,
                lane=self.lane,
                device=self.device,
                max_workspaces=self.max_workspaces,
                wrappers=weakref.WeakSet(),
            )
            self.backend._fast_cp_wrapper_state = state
        if state.device != self.device or state.lane != self.lane:
            raise ValueError(
                "shared backend requires the same device and construction stream"
            )
        self._lock, self._workspaces = state.lock, state.workspaces
        self.max_workspaces = state.max_workspaces
        self.pool_bytes = self.backend.pool_bytes
        state.wrappers.add(self)

    def _probe_device(self):
        if self.device.type != "cuda" or self.world not in self.profile.cp_degrees:
            return "H20 Fast CP requires CUDA and CP2/4/8"
        from ...interconnect import local_interconnect

        p = torch.cuda.get_device_properties(self.device)
        if not self.profile.supports_device(
            p.name,
            (p.major, p.minor),
            p.multi_processor_count,
            local_interconnect().nvlink,
        ):
            return "device/interconnect does not match the H20 NVLink profile"
        return None

    @property
    def backend(self):
        if self._backend is None:
            raise RuntimeError("backend is initialized during collective prepare")
        return self._backend

    @property
    def workspace_count(self):
        return len(self._workspaces)

    def _ensure_backend(self):
        global _owner
        if self._backend is not None:
            return
        previous = _owner() if _owner is not None else None
        if previous is not None and not previous.closed:
            raise RuntimeError(
                "an owning Fast CP runtime exists; share it or pass backend= explicitly"
            )
        from .runtime.backend import FastUlyssesAllToAll

        self._backend = FastUlyssesAllToAll(
            self.process_group, self.device, pool_bytes=self.pool_bytes
        )
        self._owns_backend = True
        self._attach_backend()
        _owner = weakref.ref(self)

    def _agree(self, record):
        if self.world == 1:
            return [record]
        records = [None] * self.world
        dist.all_gather_object(records, record, group=self.process_group)
        return records

    def _check_context(self):
        if self.closed:
            raise RuntimeError("Fast CP runtime is closed")
        if not torch.is_inference_mode_enabled():
            raise RuntimeError("Fast CP requires torch.inference_mode()")
        if self.lane is not None:
            if torch.cuda.current_device() != self.device.index:
                raise RuntimeError("Fast CP requires its construction device")
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Fast CP uses segmented graphs; outer graph capture is unsupported"
                )
            current = (
                self._raw_stream(self.device.index)
                if self._raw_stream is not None
                else torch.cuda.current_stream(self.device).cuda_stream
            )
            if current != self._lane_handle:
                raise RuntimeError("Fast CP requires its construction stream")

    @contextmanager
    def _execution(self):
        self._check_context()
        if not self._lock.acquire(blocking=False):
            raise RuntimeError(
                "Fast CP workspace is already in use; concurrent execution is unsupported"
            )
        try:
            yield
        finally:
            self._lock.release()

    def wrap_attention(
        self, module, *, adapter, origin=None, algorithm="agkv", schedule="auto"
    ):
        if self.closed:
            raise RuntimeError("Fast CP runtime is closed")
        if algorithm not in {"agkv", "ulysses"} or schedule not in {
            "auto",
            "packed",
            "overlap",
            "serial",
        }:
            raise ValueError("invalid algorithm or execution schedule")
        if algorithm == "ulysses" and schedule not in {"auto", "serial"}:
            raise ValueError("Ulysses currently uses the serial template")
        if algorithm == "agkv" and schedule == "serial":
            raise ValueError("use packed or overlap for AGKV")
        block = FastCPAttention(
            self, module, adapter, origin, algorithm, schedule, len(self._blocks)
        )
        self._blocks.append(block)
        return block

    def parallel(self, tag):
        from ..topology import UlyssesTopology

        transport = _SerialUlyssesTransport(self.backend, tag)
        topology = UlyssesTopology(
            tuple(range(self.world)),
            self.rank,
            self.world,
            self.process_group,
            transport,
        )
        return SimpleNamespace(
            active=SimpleNamespace(
                width=self.world,
                rank_in_lane=self.rank,
                process_group=self.process_group,
            ),
            active_ulysses=topology,
            active_agkv_transport=None,
        )

    def _prepare(self, block, inputs, bound):
        # All Python capability errors are collected before allocating a heap or
        # invoking a model builder with collectives. Runtime failures are fatal.
        error, signature, reason = None, None, self._device_reason
        transport_config = None
        try:
            self._check_context()
            from .runtime.backend import _ulysses_transfer_from_env, _use_tma_from_env

            transport_config = (
                self._backend._transfer if self._backend is not None else None,
                _ulysses_transfer_from_env(),
                _use_tma_from_env(),
            )
            signature = (
                block.index,
                block.generation,
                block.algorithm,
                block.schedule,
                block.adapter.signature(),
                _signature(inputs),
                _module_signature(block.module),
                self.profile.name,
                self.profile.version,
                self.pool_bytes,
                self.max_workspaces,
                bound,
            )
            if not reason:
                previous = _owner() if _owner is not None else None
                if (
                    self._backend is None
                    and previous is not None
                    and not previous.closed
                ):
                    raise ValueError(
                        "an owning Fast CP runtime exists; share it or pass backend= explicitly"
                    )
                from ...tp.topology import get_tp_world_size

                if get_tp_world_size() != 1:
                    raise UnsupportedFastCP("wrapper currently supports TP1 only")
                tensors = []
                _map(inputs, lambda t: tensors.append(t))
                if any(t.device != self.device for t in tensors):
                    raise UnsupportedFastCP(
                        "all region inputs must be on the runtime device"
                    )
                if inputs["hidden_states"].dtype != torch.bfloat16:
                    raise UnsupportedFastCP(
                        "current H20 planner calibration requires BF16"
                    )
                block.adapter.validate(block.module, inputs, block.algorithm)
                from .runtime.backend import probe_fast_ulysses

                available, detail = probe_fast_ulysses(
                    self.device, require_hopper=True, require_subgroups=False
                )
                if not available:
                    raise UnsupportedFastCP(detail)
                if block.algorithm == "agkv":
                    if not all(
                        hasattr(torch.ops.fast_ulysses, name)
                        for name in (
                            "kv_projection_prepare",
                            "projection_sync_prepare",
                            "kv_projection_send_ce_group",
                        )
                    ):
                        raise UnsupportedFastCP(
                            "extension lacks the direct-slot KV projection operators"
                        )
                    from .attention.cute import load_cute_flash_attn_fwd

                    load_cute_flash_attn_fwd()
        except UnsupportedFastCP as exc:
            reason = str(exc)
        except (ImportError, OSError) as exc:
            reason = str(exc)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        records = self._agree(
            (
                signature,
                error,
                reason,
                block.origin is not None,
                socket.gethostname(),
                tuple(sorted(self._workspaces)),
                self._backend is not None,
                transport_config,
            )
        )
        errors = [r[1] for r in records if r[1]]
        if errors:
            raise ValueError(f"Fast CP preparation failed collectively: {errors}")
        if any(r[0] != signature for r in records):
            raise ValueError("Fast CP preparation signatures differ across ranks")
        if any(r[5] != records[0][5] for r in records):
            raise ValueError("Fast CP workspace registries differ across ranks")
        if any(r[6] != records[0][6] for r in records):
            raise ValueError("Fast CP backend initialization differs across ranks")
        if any(r[7] != records[0][7] for r in records):
            raise ValueError("Fast CP transfer configuration differs across ranks")
        reasons = sorted({r[2] for r in records if r[2]})
        if len({r[4] for r in records}) != 1:
            reasons.append("Fast CP requires a node-local group")
        if reasons:
            if not all(r[3] for r in records):
                raise UnsupportedFastCP("; ".join(reasons))
            return _Prepared(
                block,
                inputs,
                None,
                bound,
                {"path": "origin", "reason": "; ".join(reasons)},
            )

        # Deliberately omit layer identity and weight addresses: sequential layers
        # with the same geometry reuse the same symmetric buffers. Graphs/weights
        # remain per-layer. Max unique geometries bounds monotonic heap growth.
        # Use a stable geometry hash, independent of the order keys were registered.
        geometry = hashlib.sha256(repr(signature[2:-1]).encode()).hexdigest()[:20]
        if (
            geometry not in self._workspaces
            and len(self._workspaces) >= self.max_workspaces
        ):
            raise RuntimeError(
                "Fast CP workspace limit reached; create a new runtime for new shapes"
            )
        self._ensure_backend()
        tag = self._workspaces.setdefault(geometry, f"fastcp_{geometry}")
        with self._execution():
            if self.lane is not None:
                self.lane.synchronize()  # Preparation may reset reused barrier state.
            stable = inputs if bound else _map(inputs, lambda t: t.clone())
            execution = block.adapter.build(
                self, block.module, stable, block.algorithm, block.schedule, tag
            )
            execution(stable)  # Complete transfer selection and attention preparation.
            metadata = dict(
                execution.describe(),
                path="fast",
                algorithm=block.algorithm,
                hardware_profile=self.profile.name,
                workspace=tag,
                input_policy="bound" if bound else "dynamic",
                selected_transfers=self.backend.selected_transfers,
            )
        return _Prepared(block, stable, execution, bound, metadata)

    def close(self):
        if self.closed:
            return
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("cannot close Fast CP during execution")
        try:
            state = getattr(self._backend, "_fast_cp_wrapper_state", None)
            if self._owns_backend and any(
                w is not self and not w.closed for w in state.wrappers
            ):
                raise RuntimeError(
                    "close borrowed wrappers before closing the owning runtime"
                )
            for block in self._blocks:
                block.close()
            self._blocks.clear()
            if self._owns_backend:
                self.backend.destroy()
                self._workspaces.clear()
            if state is not None:
                state.wrappers.discard(self)
            self._backend = None
            self.closed = True
        finally:
            self._lock.release()

    def __enter__(self):
        if self.closed:
            raise RuntimeError("Fast CP runtime is closed")
        return self

    def __exit__(self, *exc):
        self.close()


class _Prepared:
    def __init__(self, block, inputs, execution, bound, metadata):
        self.block, self.inputs, self.execution = block, inputs, execution
        self.bound, self.metadata = bound, metadata
        self.generation = block.generation
        self.signature = _signature(inputs)
        self.closed = False

    def _run(self, inputs=None, *, owned=False):
        block = self.block
        if self.closed or self.generation != block.generation or block.closed:
            raise RuntimeError(
                "Fast CP execution is closed or invalidated; prepare again"
            )
        if block.module.training:
            raise RuntimeError(
                "Fast CP requires eval weights; invalidate before changing the module"
            )
        runtime = block.runtime
        runtime._check_context()
        if not runtime._lock.acquire(blocking=False):
            raise RuntimeError(
                "Fast CP workspace is already in use; concurrent execution is unsupported"
            )
        try:
            if inputs is not None:
                if _signature(inputs) != self.signature:
                    raise ValueError(
                        "input metadata changed; all ranks must prepare again before forward"
                    )
                if self.execution is not None:
                    _copy_into(self.inputs, inputs)
            if self.execution is None:
                result = block.origin(**(self.inputs if inputs is None else inputs))
            else:
                result = self.execution(self.inputs)
            return _map(result, lambda t: t.clone()) if owned else result
        finally:
            runtime._lock.release()

    def run(self):
        """Run bound inputs; returned tensors are borrowed until the next call."""
        return self._run()

    def close(self):
        if self.closed:
            return
        if self.execution is not None:
            self.execution.close()
        self.execution = None
        self.inputs = None
        self.closed = True


class FastCPAttention:
    def __init__(self, runtime, module, adapter, origin, algorithm, schedule, index):
        self.runtime, self.module, self.adapter, self.origin = (
            runtime,
            module,
            adapter,
            origin,
        )
        self.algorithm, self.schedule, self.index = algorithm, schedule, index
        self.generation, self.closed = 0, False
        self._prepared = None
        self._bindings = []

    def prepare(self, example_inputs, *, input_policy="dynamic"):
        """Collectively prepare owned input/output semantics; no forward tuning."""
        if input_policy != "dynamic":
            raise ValueError("use bind(inputs=...) for fixed-address execution")
        if self.closed:
            raise RuntimeError("Fast CP region is closed")
        prepared = self.runtime._prepare(self, dict(example_inputs), False)
        if self._prepared is not None:
            self._prepared.close()
        self._prepared = prepared
        return self

    def bind(self, *, inputs):
        """Collectively bind fixed tensor addresses; values may change in place."""
        if self.closed:
            raise RuntimeError("Fast CP region is closed")
        prepared = self.runtime._prepare(self, dict(inputs), True)
        self._bindings.append(prepared)
        return prepared

    @property
    def metadata(self):
        return None if self._prepared is None else self._prepared.metadata

    def __call__(self, **inputs):
        if self._prepared is None:
            raise RuntimeError(
                "call prepare(example_inputs=...) collectively before forward"
            )
        # Even an origin callable may expose scratch. Clone under the runtime
        # lease, before another call can enqueue writes to a shared workspace.
        return self._prepared._run(inputs, owned=True)

    def invalidate(self):
        """Call on every rank before weight/LoRA changes, then prepare again."""
        self.generation += 1
        if self._prepared is not None:
            self._prepared.close()
            self._prepared = None
        for binding in self._bindings:
            binding.close()
        self._bindings.clear()

    def close(self):
        if not self.closed:
            self.invalidate()
            self.closed = True


def init_fast_cp(process_group=None, **kwargs):
    """Create a lazy context; pass backend= to borrow an existing NVSHMEM heap."""
    return FastCPRuntime(process_group, **kwargs)
