# CP/DP Hotswitch + Continuous Batching — Session Summary (2026-07-05)

## TL;DR

Today we built and independently verified **most of the mechanism layer** for a
step-level DiT serving scheduler (cost model, work-item abstraction, continuous
batching, mixed-step batching, dynamic SP-degree switching, a Poisson serving
harness, and profilers). **What we have NOT done is connect them into one closed
loop** driven by a load-aware policy — today every mechanism is exercised by a
*static config flag*, not by a scheduler that reads system state and chooses
grouping / SP-degree / cache per step. The scaffolds are real and each is
correct in isolation; the "string them into a payoff" step is exactly what's
left. Nothing is committed yet.

The other honest takeaway: on **NVLink, Z-Image is compute-bound end-to-end**, so
the parallelism knobs we built are a **latency-under-load** lever, not a
throughput lever. The genuine throughput/SLO lever on NVLink is **FlexCache**
(compute reduction), whose *scheduling* value (1+1>2) is still unproven and
should be settled by simulation before we build it.

---

## 1. What we built today (the scaffolds)

State legend: ✅ verified · ⚠️ partial/limited · ✋ uncommitted (everything is uncommitted).

### A. Cost model & measurement (M1) ✅
- `chitu_diffusion/runtime/cost_model.py` — decoupled **worker roofline** (compute
  latency vs batch×seq_len) + **analytical comm model** (NVLink all-to-all).
- `experiments/cp_dp_hot_switch/profile_worker.py` — GPU profiler that populates the grids.
- `experiments/cp_dp_hot_switch/cost_model.json` — calibrated data (comm ~123 GB/s).
- `test/test_cost_model.py` — held-out interpolation <5%, monotonicity, JSON round-trip.
- Report: `m1_cost_model_report.md`.
- **Purpose in the loop:** the scheduler's cost estimator. **Currently unused by any policy.**

### B. Work-item abstraction (M2) ✅
- `chitu_diffusion/runtime/task.py` — `WorkItem((request, sample))`; an `n_sample`
  request expands into N work-items sharing text embeddings; pool bookkeeping.
- `test/test_work_item.py`.
- **Purpose:** the granularity the scheduler assigns. Wired into the engine; scheduler still assigns per-task, not per-work-item across requests.

### C. Continuous batching engine (M3 same-step, M4 mixed-step) ✅ (with caveat)
- `chitu_diffusion/runtime/generator.py`, `scheduler.py`, `main.py`,
  `adapter/z_image.py` — `continuous_batch` (same-shape, same-step group in one
  forward) and the M4 **step-granular** path: per-item timestep unlocked, new
  requests admitted at step boundaries.
- Config: `infer.diffusion.continuous_batch`, `max_batch_items`, `step_interleave`.
- Tests: `test/test_continuous_batch_sched.py`, `test/test_z_image_mixed_forward.py`,
  plus the archived validation notes in `m4_mixed_step_report.md`.
- **Root-cause result:** mixed-step batching is **feasible** — the earlier
  "infeasible" call was pre-existing fp/kernel batch-nondeterminism (Category A),
  not architectural coupling. Per-item timestep math is bitwise-invariant.
- **Caveat:** gated by static flags; correctness verified, but the *throughput*
  measured is ~flat at 1024² (compute-bound) — it's a latency/queue lever.

### D. Dynamic SP-degree switching (M6) ✅
- `chitu_diffusion/runtime/sp_migration.py`, `hotswitch.py`;
  `core/distributed/parallel_state.py` (pre-warmed groups).
- Tests: `test/test_sp_migration.py`; archived multi-GPU validation details live in
  `m6_dynamic_sp_report.md`.
- **Result:** mid-run SP1→2 / SP2→4 switch is **bitwise-identical** to fixed
  degree; switch cost ~**17 µs** (pre-warmed groups → O(1) pointer swap, 0 bytes
  for Z-Image's replicated latent, 0 `new_group` on the hot path).
- **Purpose:** the parallelism actuator the scheduler will call. **Currently driven by a fixed schedule in a test, not a policy.**

### E. Serving harness — Poisson client + server replayer ⚠️
- `experiments/cp_dp_hot_switch/gen_client_trace.py` — Poisson arrival trace
  generator (rate, mixed shapes, n_sample dist, real prompts).
- `test/test_z_image_serve.py` + `test/configs/z_image_serve*.yaml` — arrival-timed
  replayer: injects each request at its `arrival_ms`, records end-to-end
  latency / queue delay / admission delay / throughput → `serve_metrics.json`.
- `traces/serve/*.json`.
- **Limitation:** **world_size==1 only** (single GPU). Multi-rank idle-sync
  between arrivals is not implemented, so we **cannot yet run a multi-GPU dynamic
  scheduler end-to-end under a trace** — a key missing bridge.

### F. Profilers & simulator ✅
- `experiments/cp_dp_hot_switch/profile_stages.py` + `stage_profile_report.md` —
  text-encode / VAE / save are **also compute-bound**; batching them is a no-go;
  only async image-save overlap is worthwhile.
- `simulate.py`, `run_experiment.py`, `traces/*` — offline policy simulator (predates today, still abstract-profile-based).

---

## 2. Key empirical conclusions

1. **NVLink Z-Image is compute-bound throughout** (M1). Batching doesn't cut FLOPs.
2. **Batching (M3/M4) = latency/queue lever, not throughput.** Staggered 1024²/20-step/12-req: throughput ~flat (+3%), queue delay −68%, mean latency can *worsen* under naive same-shape coupling (batch-coupled completions).
3. **SP is a latency lever** (~3.2× on 4 GPU) and **dynamic SP switching is exact + effectively free** (M6).
4. **Non-denoise stages are compute-bound too** — no batch benefit; async save overlap is the only non-denoise win.
5. **Group-free / symmetric-memory collectives: decided NO-GO.** Pre-warmed NCCL groups already give dynamic switching everywhere; SM-free comm is useless on NVLink (compute-bound) and unavailable on PCIe (no P2P → host bounce). Revisit only for multi-node + arbitrary membership on an NVSwitch/NVSHMEM fabric.
6. **Launcher gotcha (important for all runs):** `chitu` console script imports the
   **main-repo** copy. To run worktree code use
   `cd <worktree> && CHITU_PYTHON_BIN=… python -m chitu_diffusion.cli run <config>`.

---

## 3. Proven mechanisms vs. the missing connective tissue

**Proven in isolation (the actuators exist and are correct):**
- expand request → work-items (M2)
- batch same-shape work-items in one forward (M3)
- batch *different-step* work-items + admit at step boundary (M4)
- switch SP degree mid-run, bitwise, ~free (M6)
- estimate step cost from batch×seq_len (M1)
- drive a Poisson trace and measure SLO metrics (harness, 1-GPU)

**Not yet connected — this is the whole "string it together" gap:**
- ❌ **No policy uses the cost model.** M7 (the load-aware scheduler that picks
  grouping + SP-degree per step to optimize throughput/SLO) is not built. Every
  mechanism is toggled by a static flag today.
- ❌ **No multi-GPU end-to-end serving loop.** The serve harness is 1-GPU; the
  dynamic-SP switching (M6) is multi-GPU but driven by a scripted schedule.
  Nothing yet joins {Poisson trace → scheduler(cost model) → {batch, SP-degree}
  → metrics} on ≥2 GPUs.
- ❌ **FlexCache is not wired** as a scheduling lever, and its superadditivity is unproven.
- ❌ **Nothing committed** — the whole feature set is an uncommitted working tree
  (16 modified + many new files).

Intended closed loop (target), vs. today (pieces):

```
   TARGET:  trace ─▶ scheduler ──(cost model)──▶ choose {group size, SP degree, cache}
                        ▲                                   │
                        └────────── metrics ◀───── execute (batch + SP-switch) ─┘

   TODAY:   [trace]      [scheduler: FIFO + static flags]      [cost model: standalone]
            [batch ✔]    [SP-switch ✔]    [serve metrics ✔, 1-GPU]    [FlexCache: not wired]
            ── all present, none wired through a policy ──
```

---

## 4. The benefit thesis (honest, for tomorrow's design)

- **Parallelism (CP↔DP) is a latency-under-load lever.** Real on NVLink at low
  load (~3.2× single-request), throughput-neutral at saturation. Value = riding
  the latency↔throughput Pareto as load varies (CP when idle, DP when busy).
- **Throughput/SLO win needs a compute-reduction lever**, i.e. **FlexCache** on
  NVLink (or PCIe, where CP→DP is itself a throughput move because CP is comm-bound).
- **The right metric is 2-D:** quality × SLO-attainment across a *load sweep*
  under bursty (Poisson) arrivals — not a single steady-state throughput number.
- **FlexCache-as-scheduling (1+1>2) is conditional and must be proven:** it only
  beats "just always cache" if caching is genuinely lossy AND the optimal cache
  level is load/deadline/priority-dependent. Falsifiable; test before building.

---

## 5. Open design questions for tomorrow

1. **M7 — the load-aware scheduler.** Make grouping + SP-degree (and later cache)
   the scheduler's action space, use `cost_model.py` to estimate, optimize a
   throughput/SLO objective. Prove it beats the *best static* policy first in the
   simulator, then on real GPUs.
2. **Multi-GPU serving loop.** Add the idle-sync barrier so the serve harness runs
   at world_size>1 — the bridge that lets M6 switching run under a live trace.
3. **FlexCache superadditivity study** (4 policies A/B/C/D on a bursty SLO trace,
   cost model + a quality-degradation model) — go/no-go before implementing.
4. **Target-hardware decision.** NVLink → pitch as latency/SLO (parallelism) +
   throughput (FlexCache); PCIe → parallelism is also a throughput story. This
   choice reframes the whole evaluation.
5. **Commit strategy.** Decide how to land the M1–M6 scaffolds (categorized
   commits) so tomorrow's integration work has a clean base.

---

## 6. Milestone tracker

| ID | Milestone | Status |
|---|---|---|
| M1 | Worker roofline + comm model | ✅ done |
| M2 | Work-item ((request, sample)) | ✅ done |
| M3 | Same-shape same-step continuous batching | ✅ done |
| M4 | Mixed-step batching + step-boundary admit (feasible; latency lever) | ✅ done |
| M5 | Mixed-shape `cu_seqlens` varlen packing | ⏸ optional, not started |
| M6 | Dynamic SP-degree switch + pre-warmed groups (bitwise, ~17 µs) | ✅ done |
| M7 | Load-aware throughput/SLO scheduler (cost-model driven) | ⛔ not started — **the integration** |
| — | FlexCache-as-scheduling-lever | ❓ unproven — superadditivity study first |
