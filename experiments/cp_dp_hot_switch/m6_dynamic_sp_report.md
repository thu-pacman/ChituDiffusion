# M6 — Dynamic Sequence-Parallel (SP/CP) Degree Switching at Step Boundaries

**Goal.** Give the M4 step-granular engine the *mechanism* to change a work-item's
SP/CP degree between denoise steps, with **zero NCCL group-creation cost on the hot
path** (pre-warmed process groups) and **bit-exact** results versus running the whole
request at a fixed degree. This is the foundation the M7 throughput scheduler will drive.

Everything is gated behind `infer.diffusion.dynamic_sp` (default **false**). With the
flag off, every M0–M4 path is byte-identical (verified below).

---

## 1. Pre-warm scheme (overlapping process groups)

At init (`initialize_diffusion_parallel_groups(..., dynamic_sp=True)` in
`chitu_diffusion/core/distributed/parallel_state.py`) we enumerate **every feasible SP
degree** given the world size and build a `CommGroup` for each *before* any request runs:

- World size `W`, base CP degree `C` (from `cfp`). Feasible degrees = all divisors `d`
  of `W` with `1 <= d <= W` that tile the ranks into contiguous blocks
  (e.g. `W=4 -> {1, 2, 4}`, `W=2 -> {1, 2}`).
- For degree `d`, ranks are split into `W/d` contiguous SP groups of size `d`
  (`[0..d), [d..2d), ...`); the complement `W/d` is the DP/replica arrangement. This
  matches the existing contiguous-block CP convention, so `ImageContextParallelAttention`
  and the CP core work unchanged on any pre-warmed group.
- All groups are cached in `_DYNAMIC_CP_GROUP_DICT[d]`. Switching degree at runtime is
  `set_active_cp_group(d)` — an **O(1) pointer swap** of the active `_CP_GROUP` plus
  `_ACTIVE_CP_DEGREE`. No `dist.new_group` is called after init.

**Verification of zero hot-path group creation:** the test driver monkey-patches
`torch.distributed.new_group` with a counter around the timed run.

```
pre-warmed degrees=[1, 2, 4] base=4        (4-GPU)
pre-warmed degrees=[1, 2]    base=2        (2-GPU)
new_group calls during hot-path run = 0    (expected 0)   ✓
```

---

## 2. Migration algorithm (latent re-shard)

`chitu_diffusion/runtime/sp_migration.py` owns the index math and collectives. A logical
sequence of length `L` is sharded contiguously across `d` ranks using the same convention
as Z-Image `cp_forward`:

```
chunk = L // d ;  rank r owns global positions [r*chunk, (r+1)*chunk)   (L % d == 0)
```

(non-divisible `L` falls back to the `torch.tensor_split` convention for the CPU tests;
the GPU path only shards when `img_len % d == 0`, else it skips CP for that step).

### (a) gather + scatter — correctness-first
`migrate_gather_scatter(local_shard, old_group, new_degree, new_rank)`: all-gather the
full sequence on the old group, then slice this rank's new shard
(`reshard_full_to_shard`). Obviously correct; moves `(d-1)/d` of the sequence to every
rank. Used as the reference and the general-case path.

### (b) slice-intersection — bandwidth-optimal (index math, P2P-ready)
`slice_intersections(L, old_d, new_d)` returns every non-empty overlap
`[max(olo,nlo), min(ohi,nhi))` between an OLD shard and a NEW shard. The union of these
transfers **exactly tiles `[0, L)`** — no gaps, no double-cover — so applying them
reconstructs the new sharding losslessly. `rank_migration_plan(...)` turns that into a
per-rank `sends` / `recvs` / `keeps` list in local shard coordinates:
- `keeps`: the overlap already on the right rank (no comm — e.g. on SP2→SP4 rank 0's new
  half stays put),
- `sends`/`recvs`: only the intersection slices exchanged with the peers whose ranges now
  overlap, ready for P2P `isend`/`irecv`.

This is the "codec + slice-intersection" plan; the P2P collective itself is the M-later
follow-up (out of scope). It is fully unit-tested (below).

### Z-Image specifics (image tokens / RoPE)
Z-Image keeps the **full, replicated latent at the task level**; sequence sharding lives
entirely *inside* `cp_forward` (image tokens are split on the token dim there, and RoPE /
position ids are recomputed per forward from the shard offset on whichever CP group is
active). So a degree switch needs **no task-level data movement** — the switch is purely
`set_active_cp_group(d)`, and the next `cp_forward` shards on the new group with correct
per-shard RoPE offsets automatically. `migrate_task_latents` is therefore a no-op for
Z-Image (`migrated_bytes=0`); the gather+scatter / slice-intersection primitives exist and
are validated for the general sharded-latent case and are exercised by the
micro-benchmark.

### Switch wiring (M4 step loop)
The switch fires at **step boundaries**: `generator._maybe_switch_sp_degree(task)` is
invoked at the top of each iteration of the step-granular denoise loop (in the Z-Image
adapter's per-step loop). It consults `_sp_target_for_step(step)` (driven by
`CHITU_SP_INITIAL/SWITCH_STEP/TARGET` for verification; M7 will own it programmatically),
and on a change: migrates latents (no-op for Z-Image), calls `set_active_cp_group(target)`,
and records switch wall time.

---

## 3. Verification

Historical launcher (worktree): `python -m chitu_diffusion.cli run <config>`, `attn_type=torch_sdpa`
(deterministic reference backend), 1024², 8 steps, seed 42, switch at step 4. The original
one-off validation used a temporary latent-dump hook to compare final latents on rank 0;
that production-coupled dump hook was removed during cleanup, so this report should be read
as the archived M6 verification result rather than a currently committed GPU test driver.

### (a) Mid-run switch correctness — **BITWISE**

4-GPU historical driver (degrees {1,2,4}):

| comparison | max_abs | rms | verdict |
|---|---|---|---|
| switch 1→2 vs fixed 1 | 0.000e+00 | 0.000e+00 | **BITWISE** |
| switch 1→2 vs fixed 2 | 0.000e+00 | 0.000e+00 | **BITWISE** |
| switch 2→4 vs fixed 2 | 0.000e+00 | 0.000e+00 | **BITWISE** |
| switch 2→4 vs fixed 4 | 0.000e+00 | 0.000e+00 | **BITWISE** |
| fixed 1 vs fixed 2 | 0.000e+00 | 0.000e+00 | **BITWISE** |
| fixed 1 vs fixed 4 | 0.000e+00 | 0.000e+00 | **BITWISE** |
| fixed 2 vs fixed 4 | 0.000e+00 | 0.000e+00 | **BITWISE** |

2-GPU historical driver (degrees {1,2}): switch 1→2 vs fixed 1 / fixed 2 and
fixed 1 vs fixed 2 all **BITWISE (0.000e+00)**.

**Exactness vs Category-A noise.** SP/ring attention is mathematically exact, and under
`torch_sdpa`/BF16 with a single request (no batch-composition change) the reduction order
is identical across degrees, so we get **bitwise 0**, not merely within-tol. This is
stronger than the ~1e-5 BF16 tolerance the M4 mixed-step gate needed (that noise came from
*batch-composition-dependent* reductions, which do not occur here — a fixed request keeps
the same per-shard reduction structure at any degree). The bitwise cross-check of the
three fixed degrees confirms SP is exact across {1,2,4} independent of the switch.

### (b) Migration / switch overhead — **negligible**

| switch | wall time | % of one denoise step | bytes moved |
|---|---|---|---|
| SP 1→2 @ step 4 (4-GPU) | 0.0189 ms | 0.0017 % (step ≈ 1102 ms) | 0 |
| SP 2→4 @ step 4 (4-GPU) | 0.0166 ms | 0.0015 % (step ≈ 1076 ms) | 0 |
| SP 1→2 @ step 4 (2-GPU) | 0.017 ms | 0.0015 % (step ≈ 1092 ms) | 0 |

The switch is a pre-warmed group-pointer swap → **0 bytes** moved for Z-Image and ~17 µs,
i.e. ~1.5×10⁻³ % of a denoise step. General-case gather+scatter re-shard collective
(sharded-latent models), micro-benchmarked on the pre-warmed groups:

| collective | wall time | % of a step |
|---|---|---|
| reshard on CP=2 | 0.1103 ms | 0.010 % |
| reshard on CP=4 | 0.1277 ms | 0.012 % |

Even the full all-gather+scatter re-shard is ~0.01 % of a step — the slice-intersection
P2P path would be strictly cheaper. **Zero `new_group` calls at switch time** (§1).

### (c) Sanity — baseline & fixed-degree unchanged

- **Baseline `dynamic_sp` OFF** (historical standard driver, fixed cp=2)
  vs M6 `fixed_d2` latent: `max_abs=0.000e+00` → **BITWISE**. The dynamic-SP code paths
  are fully inert when the flag is off (wrapper-install gate, group pre-warm, and the
  step-loop switch hook all short-circuit).
- Fixed-degree runs (degree held constant) reproduce the baseline exactly (above table).
- CPU unit tests: `test/test_continuous_batch_sched.py` (M3/M4 scheduler/mixed-step
  admission) + `test/test_sp_migration.py` (slice-intersection index math) → **18 passed**.

---

## 4. Files changed (all uncommitted)

- `chitu_diffusion/core/schemas/serve_config.py` — `dynamic_sp: bool = False` on `DiffusionConfig`.
- `chitu_diffusion/core/config/serve_config.yaml` — `dynamic_sp: false` default.
- `chitu_diffusion/core/distributed/parallel_state.py` — pre-warm all feasible CP groups
  (`initialize_dynamic_sp_groups`), `get/set_active_cp_group`, `get_dynamic_cp_group`,
  degree-aware `get_cp_size`.
- `chitu_diffusion/runtime/backend.py` — thread `dynamic_sp` into group init.
- `chitu_diffusion/runtime/adapter/z_image.py` — install CP wrapper when `dynamic_sp` even
  at cp=1; call `_maybe_switch_sp_degree` at each step boundary in the denoise loop.
- `chitu_diffusion/runtime/generator.py` — `_sp_target_for_step`, `_maybe_switch_sp_degree`,
  `_migrate_task_latents_for_switch`, env-driven switch schedule.
- `chitu_diffusion/runtime/sp_migration.py` *(new)* — `shard_bounds`, `slice_intersections`,
  `rank_migration_plan`, `migrate_gather_scatter`, reshard helpers.
- `test/test_sp_migration.py` *(new)* — CPU tests for the migration index math.
- Multi-GPU correctness was validated with a temporary one-off driver before cleanup; the
  cleaned commit keeps the CPU migration tests and the report, not the latent-dump GPU driver.

## 5. Reproduce

```bash
python -m pytest test/test_continuous_batch_sched.py test/test_sp_migration.py -q
```

---

## 6. Conclusion

- **Mid-run SP switching is correct.** SP1→2 and SP2→4 mid-run switches produce final
  latents **bitwise-identical** to fixed-degree runs under `torch_sdpa` (stronger than the
  ~1e-5 BF16 tolerance the M4 mixed-step gate allowed — no Category-A noise arises for a
  fixed single request). Baseline (flag off) and fixed-degree runs are unchanged.
- **Switch overhead is negligible.** With pre-warmed groups, a switch is an O(1)
  group-pointer swap (~17 µs, ~0.0015 % of a denoise step) and moves **0 bytes** for
  Z-Image's replicated-latent layout; even the general gather+scatter re-shard is
  ~0.01–0.012 % of a step. **Zero** group-creation on the hot path.
