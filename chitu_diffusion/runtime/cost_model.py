"""Decoupled cost model for DiT serving: worker roofline + communication.

The model deliberately separates the two things that determine a denoise step's
latency, because they scale differently and are actionable in different ways:

* **Compute (worker roofline)** -- ``RooflineComputeModel`` gives the latency of a
  *single worker* running one DiT forward as a function of the batch size ``B``
  and the *per-worker* sequence length ``L`` (image tokens after any sequence
  sharding, plus replicated text tokens). Small ``B*L`` sits in the weight-load
  (memory-bound) floor where adding batch is nearly free; large ``B*L`` sits on
  the compute-bound roofline where only reducing ``L`` (sequence parallel) helps.
  Backed by a measured ``B x L`` grid when available, with a physically-motivated
  roofline fallback.

* **Communication** -- ``CommModel`` gives the extra time a sequence-parallel
  degree ``k>1`` adds per step (Ulysses all-to-all over single-node NVLink),
  modelled analytically from message volume and calibrated bandwidth.

``predict_step_ms`` composes them: ``T_step ~= T_compute(B, L/k + L_txt) + T_comm(k)``.

This module is intentionally dependency-free (stdlib only) so the offline
simulator and the scheduler can import it without pulling in torch.
"""

from __future__ import annotations

import bisect
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def image_token_count(width: int, height: int, *, vae_downsample: int = 8, patch_size: int = 2) -> int:
    """Number of image tokens the DiT sees for a given pixel resolution.

    Z-Image / Flux-style latent pipelines downsample by ``vae_downsample`` (VAE)
    then patchify by ``patch_size``. E.g. 1024x1024 -> 128x128 latent -> 64x64 =
    4096 tokens.
    """
    lh = height // vae_downsample
    lw = width // vae_downsample
    return (lh // patch_size) * (lw // patch_size)


# --------------------------------------------------------------------------- #
# Architecture facts
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelShapeSpec:
    """Static architecture facts the analytical models need."""

    name: str
    dim: int
    n_heads: int
    head_dim: int
    n_layers: int
    text_tokens: int = 512
    dtype_bytes: int = 2  # bf16/fp16
    mlp_ratio: float = 4.0
    collectives_per_layer: int = 2  # Ulysses: all-to-all before + after attention

    def layer_flops(self, batch: int, seq_len: int) -> float:
        """Approx FLOPs for one transformer layer forward (linear + attention).

        linear (qkv + out + mlp): ~ (4 + 2*mlp_ratio) * 2 * B * L * dim^2
        attention (scores + context): ~ 4 * B * L^2 * dim
        The factor 2 converts multiply-adds to FLOPs.
        """
        linear = (4.0 + 2.0 * self.mlp_ratio) * 2.0 * batch * seq_len * self.dim * self.dim
        attn = 4.0 * batch * seq_len * seq_len * self.dim
        return linear + attn

    def forward_flops(self, batch: int, seq_len: int) -> float:
        return self.n_layers * self.layer_flops(batch, seq_len)

    def weight_bytes(self) -> float:
        """Rough parameter bytes touched per forward (batch/seq independent)."""
        per_layer_params = (4.0 + 2.0 * self.mlp_ratio) * self.dim * self.dim
        return self.n_layers * per_layer_params * self.dtype_bytes


Z_IMAGE_SPEC = ModelShapeSpec(
    name="Z-Image",
    dim=3840,
    n_heads=30,
    head_dim=128,
    n_layers=32,  # 30 main + 2 refiner
    text_tokens=512,
    dtype_bytes=2,
)


KNOWN_SPECS: dict[str, ModelShapeSpec] = {
    "Z-Image": Z_IMAGE_SPEC,
    "z-image": Z_IMAGE_SPEC,
}


# --------------------------------------------------------------------------- #
# Compute roofline
# --------------------------------------------------------------------------- #
@dataclass
class RooflineComputeModel:
    """Single-worker single-step DiT forward latency as ``f(batch, seq_len)``.

    If a measured grid is present, ``predict`` bilinearly interpolates on
    ``(batch, seq_len)`` (and clamps/linearly extrapolates at the edges). If no
    grid is present it falls back to the roofline
    ``T = max(mem_floor_ms, flops / peak_flops_per_ms)``.
    """

    spec: ModelShapeSpec
    # measured grid: {(batch, seq_len): latency_ms}
    grid: dict[tuple[int, int], float] = field(default_factory=dict)
    # roofline fallback anchors (H20/H800-class single GPU defaults; calibrate!)
    peak_tflops: float = 120.0  # effective (not spec-sheet) bf16 TFLOP/s
    mem_bandwidth_gbps: float = 1500.0  # effective HBM GB/s
    launch_floor_ms: float = 1.0

    # ---- grid access -------------------------------------------------------- #
    def _axes(self) -> tuple[list[int], list[int]]:
        batches = sorted({b for (b, _) in self.grid})
        seqs = sorted({s for (_, s) in self.grid})
        return batches, seqs

    def _interp_1d(self, xs: list[int], value: int) -> tuple[int, int, float]:
        """Return (lo, hi, frac) so result = table[lo]*(1-frac) + table[hi]*frac."""
        if value <= xs[0]:
            return xs[0], xs[0], 0.0
        if value >= xs[-1]:
            return xs[-1], xs[-1], 0.0
        i = bisect.bisect_right(xs, value)
        lo, hi = xs[i - 1], xs[i]
        frac = (value - lo) / (hi - lo) if hi > lo else 0.0
        return lo, hi, frac

    def _grid_predict(self, batch: int, seq_len: int) -> float:
        batches, seqs = self._axes()
        b_lo, b_hi, b_f = self._interp_1d(batches, batch)
        s_lo, s_hi, s_f = self._interp_1d(seqs, seq_len)

        def cell(b: int, s: int) -> float:
            if (b, s) in self.grid:
                return self.grid[(b, s)]
            # nearest fallback for ragged grids
            best = min(self.grid, key=lambda key: (abs(key[0] - b), abs(key[1] - s)))
            return self.grid[best]

        top = cell(b_lo, s_lo) * (1 - s_f) + cell(b_lo, s_hi) * s_f
        bot = cell(b_hi, s_lo) * (1 - s_f) + cell(b_hi, s_hi) * s_f
        return top * (1 - b_f) + bot * b_f

    def _roofline_predict(self, batch: int, seq_len: int) -> float:
        flops = self.spec.forward_flops(batch, seq_len)
        t_compute_ms = flops / (self.peak_tflops * 1e12) * 1e3
        t_mem_ms = self.spec.weight_bytes() / (self.mem_bandwidth_gbps * 1e9) * 1e3
        return max(t_mem_ms + self.launch_floor_ms, t_compute_ms)

    def predict(self, batch: int, seq_len: int) -> float:
        if self.grid:
            return self._grid_predict(int(batch), int(seq_len))
        return self._roofline_predict(int(batch), int(seq_len))

    # ---- roofline diagnostics ---------------------------------------------- #
    def batch_saturation(self, seq_len: int, batches: Optional[list[int]] = None, tol: float = 0.10) -> int:
        """Largest batch whose total step latency still sits within ``tol`` of B=1.

        This is the roofline knee: in the memory-bound floor the weight-load time
        dominates, so adding items barely moves total latency (batching is nearly
        free); past the knee the worker is compute-bound and total latency climbs
        with batch. The scheduler batches up to this knee before it pays to shard
        the sequence (sequence parallel) instead. Returns the last batch before
        total latency exceeds ``base*(1+tol)``.
        """
        candidates = batches or (sorted({b for (b, _) in self.grid}) or [1, 2, 4, 8, 16])
        base = self.predict(candidates[0], seq_len)
        knee = candidates[0]
        for b in candidates[1:]:
            if self.predict(b, seq_len) <= base * (1.0 + tol):
                knee = b
            else:
                break
        return knee


# --------------------------------------------------------------------------- #
# Communication model (single-node NVLink, Ulysses all-to-all)
# --------------------------------------------------------------------------- #
@dataclass
class CommModel:
    """Extra per-step latency from sequence-parallel collectives (k>1)."""

    spec: ModelShapeSpec
    nvlink_gbps: float = 300.0  # effective per-rank all-to-all bandwidth (calibrate)
    call_latency_ms: float = 0.010  # fixed per-collective launch/latency overhead

    def all_to_all_ms(self, k: int, tokens_global: int, batch: int) -> float:
        if k <= 1:
            return 0.0
        seq_local = tokens_global / k
        activation_bytes = batch * seq_local * self.spec.dim * self.spec.dtype_bytes
        moved = activation_bytes * (k - 1) / k
        return moved / (self.nvlink_gbps * 1e9) * 1e3 + self.call_latency_ms

    def step_comm_ms(self, k: int, tokens_global: int, batch: int) -> float:
        if k <= 1:
            return 0.0
        per_layer = self.spec.collectives_per_layer * self.all_to_all_ms(k, tokens_global, batch)
        return self.spec.n_layers * per_layer


# --------------------------------------------------------------------------- #
# Bundle + composition
# --------------------------------------------------------------------------- #
@dataclass
class CostModel:
    """A model's compute roofline + comm model, with (de)serialization."""

    compute: RooflineComputeModel
    comm: CommModel

    @property
    def spec(self) -> ModelShapeSpec:
        return self.compute.spec

    def predict_step_ms(self, *, img_tokens: int, batch: int = 1, sp_degree: int = 1) -> float:
        """Latency of one denoise step for a batch of same-shape work-items.

        ``img_tokens`` is the GLOBAL image token count for the shape; sequence
        parallelism shards it across ``sp_degree`` workers, text tokens are
        replicated. Comm is added only when ``sp_degree > 1``.
        """
        seq_local = self.spec.text_tokens + img_tokens // max(1, sp_degree)
        t_compute = self.compute.predict(batch, seq_local)
        t_comm = self.comm.step_comm_ms(sp_degree, img_tokens, batch)
        return t_compute + t_comm

    def predict_request_ms(self, *, img_tokens: int, num_steps: int, batch: int = 1, sp_degree: int = 1) -> float:
        return num_steps * self.predict_step_ms(img_tokens=img_tokens, batch=batch, sp_degree=sp_degree)

    # ---- serialization ------------------------------------------------------ #
    def to_dict(self) -> dict:
        return {
            "spec": asdict(self.spec),
            "compute": {
                "grid": [
                    {"batch": b, "seq_len": s, "latency_ms": ms}
                    for (b, s), ms in sorted(self.compute.grid.items())
                ],
                "peak_tflops": self.compute.peak_tflops,
                "mem_bandwidth_gbps": self.compute.mem_bandwidth_gbps,
                "launch_floor_ms": self.compute.launch_floor_ms,
            },
            "comm": {
                "nvlink_gbps": self.comm.nvlink_gbps,
                "call_latency_ms": self.comm.call_latency_ms,
            },
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def from_dict(cls, data: dict) -> "CostModel":
        spec = ModelShapeSpec(**data["spec"])
        c = data.get("compute", {})
        grid = {(int(item["batch"]), int(item["seq_len"])): float(item["latency_ms"]) for item in c.get("grid", [])}
        compute = RooflineComputeModel(
            spec=spec,
            grid=grid,
            peak_tflops=float(c.get("peak_tflops", 120.0)),
            mem_bandwidth_gbps=float(c.get("mem_bandwidth_gbps", 1500.0)),
            launch_floor_ms=float(c.get("launch_floor_ms", 1.0)),
        )
        m = data.get("comm", {})
        comm = CommModel(
            spec=spec,
            nvlink_gbps=float(m.get("nvlink_gbps", 300.0)),
            call_latency_ms=float(m.get("call_latency_ms", 0.010)),
        )
        return cls(compute=compute, comm=comm)

    @classmethod
    def load(cls, path: str | Path) -> "CostModel":
        return cls.from_dict(json.loads(Path(path).read_text()))


def default_cost_model(spec: ModelShapeSpec = Z_IMAGE_SPEC) -> CostModel:
    """Analytical-only model (no measurements). Coarse but self-consistent."""
    return CostModel(compute=RooflineComputeModel(spec=spec), comm=CommModel(spec=spec))
