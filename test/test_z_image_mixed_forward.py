"""M4 per-item-timestep isolation test (single batched forward, fixed membership).

The M4 unlock lets rows at DIFFERENT ``current_step`` (thus different timestep)
share ONE transformer forward. Correctness requires that a row's output depends
ONLY on its own (latent, timestep, conditioning) -- NOT on the timesteps of the
other rows sharing the forward.

This test proves exactly that, while holding batch membership FIXED at {A, B} so
the pre-existing Z-Image batch-size nondeterminism (a batched forward is not
bitwise-reproducible vs a solo forward, even for n_sample -- see result notes) is
held constant and thus cancels out:

  * A_ref  = row A's noise pred with (A@stepA, B@stepA)   [B at same step as A]
  * A_test = row A's noise pred with (A@stepA, B@stepB)   [B at a DIFFERENT step]
  -> A_ref must equal A_test BITWISE (B's timestep must not leak into A).

It also sanity-checks that (i) repeated identical calls are deterministic and
(ii) a row's own timestep DOES change its output (so the forward is actually
timestep-sensitive).

world_size==1 (SP=1) only. Drive with test/configs/z_image_mixed_forward.yaml.
"""

from __future__ import annotations

import gc
import logging
import os
from logging import getLogger

import torch

from chitu_diffusion.core.config_loader import load_config_from_cli
from chitu_diffusion.core.schemas import ServeConfig
from chitu_diffusion.runtime.backend import DiffusionBackend
from chitu_diffusion.runtime.main import chitu_init, warmup_diffusion_engine
from chitu_diffusion.runtime.task import (
    DiffusionTask,
    DiffusionUserParams,
    DiffusionUserRequest,
)
from run_context import DiffusionTestRunContext

logger = getLogger(__name__)

PROMPT_A = (
    'A compact workstation desk with a small sign reading "Z-Image x ChituDiffusion", '
    "soft morning light, crisp product photography, detailed cables and notebooks."
)
PROMPT_B = (
    "A serene mountain lake at dawn, mirror-still water, pine forest, volumetric fog, "
    "ultra detailed landscape photography."
)
NEG = "deformed, distorted, blurry, low quality, watermark, jpeg artifacts"


def _mk_task(args, task_id, prompt, seed):
    width, height = [
        int(v.strip())
        for v in os.getenv("CHITU_Z_IMAGE_SIZE", "1024,1024").lower().replace("x", ",").split(",", 1)
    ]
    steps = int(os.getenv("CHITU_Z_IMAGE_STEPS", str(args.models.sampler.sample_steps)))
    req = DiffusionUserRequest(
        request_id=task_id,
        params=DiffusionUserParams(
            role="mixed-fwd", prompt=prompt, negative_prompt=NEG, seed=seed, frame_num=1,
            size=(width, height), num_inference_steps=steps, n_sample=1, sample_solver="flowmatch_euler",
        ),
    )
    return DiffusionTask(task_id=task_id, req=req)


def main(args: ServeConfig):
    if args.models.name != "Z-Image":
        raise ValueError(f"expects Z-Image, got {args.models.name}.")
    if int(os.getenv("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("world_size==1 only.")
    logger.setLevel(logging.INFO)
    if os.getenv("CHITU_DISABLE_TF32") == "1":
        # Strict IEEE fp32 matmuls (default GPU fp32 uses TF32: ~10-bit mantissa).
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        logger.info("[mixed-fwd] TF32 disabled (strict fp32 matmul).")
    args.models.sampler.sample_steps = int(os.getenv("CHITU_Z_IMAGE_STEPS", str(args.models.sampler.sample_steps)))

    run_context = DiffusionTestRunContext(args, logger, per_task_results=True, capture_stdio_log=False)
    run_dir = run_context.build_initial_run_output_dir()
    run_context.activate_run(run_dir)

    try:
        chitu_init(args, logging_level=logging.INFO)
        torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
        gen = DiffusionBackend.generator
        adapter = DiffusionBackend.model_adapter
        if not getattr(gen, "continuous_batch", False):
            raise RuntimeError("requires infer.diffusion.continuous_batch=true")
        run_context.activate_run(run_context.build_run_output_dir([]))
        warmup_diffusion_engine(args)

        total = int(args.models.sampler.sample_steps)
        step_a = min(2, total - 1)
        step_b_same = step_a
        step_b_diff = min(step_a + 4, total - 1)

        # Build two denoise-ready work-items (text-encode + prepare) at step 0.
        task_a = _mk_task(args, "fwd_a", PROMPT_A, 42)
        task_b = _mk_task(args, "fwd_b", PROMPT_B, 43)
        gen.cb_group = []
        gen._cb_admit_one(task_a)
        gen._cb_admit_one(task_b)

        @torch.no_grad()
        def noise_preds(sa: int, sb: int):
            task_a.buffer.current_step = sa
            task_b.buffer.current_step = sb
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = adapter._z_guided_noise_pred_batch_mixed([task_a, task_b], gen, DiffusionBackend)
            return out[0].detach().float().cpu().clone(), out[1].detach().float().cpu().clone()

        a_ref, b_ref = noise_preds(step_a, step_b_same)      # A@sa, B@sa
        a_ref2, b_ref2 = noise_preds(step_a, step_b_same)    # determinism control
        a_test, b_test = noise_preds(step_a, step_b_diff)    # A@sa, B@sb_diff (B moved)
        a_move, _ = noise_preds(step_a + 1, step_b_same)     # A moved (own step matters)

        # Batch-SIZE reproducibility probe: row A alone (batch 1) vs A in {A,B}
        # (batch 2), A's inputs byte-identical. Isolates pure model batch-size
        # behavior from any RNG/initial-noise confound in the n_sample comparison.
        @torch.no_grad()
        def noise_pred_solo(sa: int):
            task_a.buffer.current_step = sa
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = adapter._z_guided_noise_pred_batch_mixed([task_a], gen, DiffusionBackend)
            return out[0].detach().float().cpu().clone()

        a_solo = noise_pred_solo(step_a)
        a_batchsize_max = (a_ref - a_solo).abs().max().item()
        a_batchsize_rel = a_batchsize_max / (a_ref.abs().max().item() + 1e-12)

        # UNIFORM-LENGTH batch-size probe: A alone (batch 1) vs [A, A2] where A2 has
        # the SAME prompt as A (identical text-token length -> NO ragged padding /
        # attention mask). This isolates a pure batch-dim / reduction-order effect
        # from the prompt-length padding effect present in the A-vs-{A,B} probe.
        task_a2 = _mk_task(args, "fwd_a2", PROMPT_A, 42)
        gen._cb_admit_one(task_a2)

        @torch.no_grad()
        def noise_pred_uniform(sa: int):
            task_a.buffer.current_step = sa
            task_a2.buffer.current_step = sa
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = adapter._z_guided_noise_pred_batch_mixed([task_a, task_a2], gen, DiffusionBackend)
            return out[0].detach().float().cpu().clone()

        a_uniform = noise_pred_uniform(step_a)
        a_uniform_max = (a_solo - a_uniform).abs().max().item()
        a_uniform_rel = a_uniform_max / (a_solo.abs().max().item() + 1e-12)

        def _dist(x: torch.Tensor, y: torch.Tensor) -> str:
            d = (x - y).abs()
            rms = d.pow(2).mean().sqrt().item()
            sig_rms = x.pow(2).mean().sqrt().item()
            frac = (d > 1e-2).float().mean().item()
            return (
                "max=%.3e mean=%.3e rms=%.3e p99=%.3e | signal_rms=%.3e rel_rms=%.3e | frac(|d|>1e-2)=%.4f"
                % (
                    d.max().item(), d.mean().item(), rms, torch.quantile(d.flatten().float(), 0.99).item(),
                    sig_rms, rms / (sig_rms + 1e-12), frac,
                )
            )

        dist_uniform = _dist(a_solo, a_uniform)
        dist_ragged = _dist(a_ref, a_solo)

        # Signed-mean bias: reduction-order fp noise is ~zero-mean (random sign); a
        # systematic bug/coupling produces a biased (nonzero-mean) difference.
        signed_mean = (a_solo - a_uniform).mean().item()
        abs_mean = (a_solo - a_uniform).abs().mean().item()

        # FP32 DECISIVE TEST: rerun the batch-size probe with the transformer in
        # float32. bf16 reduction-order noise collapses to ~1e-6 in fp32; a genuine
        # batch-dim coupling bug would persist. Cast weights up, rerun, restore.
        fp32_uniform_max = fp32_uniform_rms = float("nan")
        fp32_ragged_max = fp32_ragged_rms = float("nan")
        try:
            orig_dtype = adapter.transformer.dtype
            adapter.transformer.float()
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
                task_a.buffer.current_step = step_a
                s32 = adapter._z_guided_noise_pred_batch_mixed([task_a], gen, DiffusionBackend)[0].detach().float().cpu()
                task_a.buffer.current_step = step_a
                task_a2.buffer.current_step = step_a
                u32 = adapter._z_guided_noise_pred_batch_mixed([task_a, task_a2], gen, DiffusionBackend)[0].detach().float().cpu()
                task_a.buffer.current_step = step_a
                task_b.buffer.current_step = step_a
                r32 = adapter._z_guided_noise_pred_batch_mixed([task_a, task_b], gen, DiffusionBackend)[0].detach().float().cpu()
            fp32_uniform_max = (s32 - u32).abs().max().item()
            fp32_uniform_rms = (s32 - u32).pow(2).mean().sqrt().item()
            fp32_ragged_max = (s32 - r32).abs().max().item()
            fp32_ragged_rms = (s32 - r32).pow(2).mean().sqrt().item()
        finally:
            adapter.transformer.to(orig_dtype)

        det_ok = torch.equal(a_ref, a_ref2) and torch.equal(b_ref, b_ref2)
        # Core M4 invariant: A unchanged when only B's timestep changes.
        a_iso_max = (a_ref - a_test).abs().max().item()
        a_iso_ok = torch.equal(a_ref, a_test)
        # B's row DID change (its own step changed step_b_same -> step_b_diff).
        b_changed_max = (b_ref - b_test).abs().max().item()
        # A's row DID change when A's own step changed (sensitivity control).
        a_self_max = (a_ref - a_move).abs().max().item()

        logger.info("=" * 72)
        logger.info("[mixed-fwd] fixed membership {A,B}; A@%d, vary B step %d->%d", step_a, step_b_same, step_b_diff)
        logger.info("[mixed-fwd] determinism (repeat identical call): %s", "OK" if det_ok else "FAIL")
        logger.info(
            "[mixed-fwd] A invariant to B's timestep: %s (max_abs_diff=%.3e)",
            "OK (bitwise)" if a_iso_ok else ("OK(<1e-5)" if a_iso_max <= 1e-5 else "FAIL"),
            a_iso_max,
        )
        logger.info("[mixed-fwd] B row changed when B step changed (want >0): max_abs_diff=%.3e", b_changed_max)
        logger.info("[mixed-fwd] A row changed when A step changed (want >0): max_abs_diff=%.3e", a_self_max)
        logger.info(
            "[mixed-fwd] batch-SIZE probe RAGGED (A batch1 vs A in {A,B} diff-prompt): "
            "max_abs_diff=%.3e rel=%.3e",
            a_batchsize_max, a_batchsize_rel,
        )
        logger.info(
            "[mixed-fwd] batch-SIZE probe UNIFORM (A batch1 vs A in {A,A2} same-prompt): "
            "max_abs_diff=%.3e rel=%.3e",
            a_uniform_max, a_uniform_rel,
        )
        logger.info("[mixed-fwd] UNIFORM batch-size dist: %s", dist_uniform)
        logger.info("[mixed-fwd] RAGGED  batch-size dist: %s", dist_ragged)
        logger.info(
            "[mixed-fwd] signed-mean=%.3e abs-mean=%.3e ratio(signed/abs)=%.3f (%s)",
            signed_mean, abs_mean, signed_mean / (abs_mean + 1e-12),
            "zero-mean -> reduction-order" if abs(signed_mean) < 0.2 * abs_mean else "biased -> systematic",
        )
        logger.info(
            "[mixed-fwd] FP32 batch-size probe UNIFORM: max_abs_diff=%.3e rms=%.3e (%s)",
            fp32_uniform_max, fp32_uniform_rms,
            "bf16 reduction-order (category A: FEASIBLE)" if fp32_uniform_max <= 1e-3
            else "persists in fp32 -> genuine coupling (category B/C)",
        )
        logger.info(
            "[mixed-fwd] FP32 batch-size probe RAGGED (diff-len prompt): max_abs_diff=%.3e rms=%.3e (%s)",
            fp32_ragged_max, fp32_ragged_rms,
            "reduction-order" if fp32_ragged_max <= 1e-3
            else "PERSISTS in fp32 -> ragged padding/mask coupling (category B: fixable)",
        )

        # TRAJECTORY probe: drive A alone for `total` steps vs A batched with B after
        # `stag` steps (B admitted at step 0), advancing via the REAL M4 engine step
        # (denoise_step_group_once). Report A's per-step latent divergence to localize
        # whether it grows gradually (reduction-order) or jumps (staggered-path bug).
        stag = min(3, total - 1)
        ta_solo = _mk_task(args, "traj_a_solo", PROMPT_A, 42)
        ta_mix = _mk_task(args, "traj_a_mix", PROMPT_A, 42)
        tb_mix = _mk_task(args, "traj_b_mix", PROMPT_B, 43)
        for t in (ta_solo, ta_mix, tb_mix):
            gen.cb_group = []
            gen._cb_admit_one(t)
        init_diff = (ta_solo.buffer.latents.float() - ta_mix.buffer.latents.float()).abs().max().item()
        logger.info("[mixed-fwd][traj] initial A latent diff solo-vs-mix=%.3e (want 0)", init_diff)

        @torch.no_grad()
        def _one(gr):
            gen.cb_group = list(gr)
            adapter.denoise_step_group_once(list(gr), gen, DiffusionBackend, gen._run_dit_forward)

        for s in range(total):
            grp = [ta_mix] if s < stag else [ta_mix, tb_mix]
            # At the FIRST batched step, A's input latent is still identical to solo
            # (steps 0..stag-1 were bitwise). Measure A's noise_pred diff directly to
            # isolate the batch-composition effect on the forward from scheduler/step.
            if s == stag:
                in_diff = (ta_solo.buffer.latents.float() - ta_mix.buffer.latents.float()).abs().max().item()
                with torch.no_grad():
                    np_solo = adapter._z_guided_noise_pred_batch_mixed([ta_solo], gen, DiffusionBackend)[0].float()
                    # B at a DIFFERENT step (0) than A (stag) -- the mixed-step case.
                    tb_mix.buffer.current_step = 0
                    np_mix0 = adapter._z_guided_noise_pred_batch_mixed([ta_mix, tb_mix], gen, DiffusionBackend)[0].float()
                    # B at the SAME step as A -- isolates pure batch-size effect.
                    tb_mix.buffer.current_step = s
                    np_mixS = adapter._z_guided_noise_pred_batch_mixed([ta_mix, tb_mix], gen, DiffusionBackend)[0].float()
                    tb_mix.buffer.current_step = 0  # restore for the real staggered run
                # A batched with an IDENTICAL copy of itself (ta_solo == ta_mix here).
                # If A's row still diverges from solo -> content-independent (kernel);
                # if it matches solo -> A depends on B's CONTENT (joint attention).
                with torch.no_grad():
                    np_AA = adapter._z_guided_noise_pred_batch_mixed([ta_mix, ta_solo], gen, DiffusionBackend)[0].float()
                d_AA = (np_solo - np_AA).abs()
                d_bs = (np_solo - np_mixS).abs()      # batch-size only (B same step)
                d_full = (np_solo - np_mix0).abs()     # batch-size + B mixed step
                d_bstep = (np_mixS - np_mix0).abs()    # pure effect of B's step on A (want 0)
                logger.info("[mixed-fwd][traj] @first-batched-step input_latent_diff=%.3e", in_diff)
                logger.info("[mixed-fwd][traj]   A: solo vs [A,A-copy] (identical B content): max=%.3e rms=%.3e", d_AA.max().item(), d_AA.pow(2).mean().sqrt().item())
                logger.info("[mixed-fwd][traj]   A: solo vs [A,B@same]  (batch-size): max=%.3e rms=%.3e", d_bs.max().item(), d_bs.pow(2).mean().sqrt().item())
                logger.info("[mixed-fwd][traj]   A: solo vs [A,B@0]     (bs+mixstep): max=%.3e rms=%.3e", d_full.max().item(), d_full.pow(2).mean().sqrt().item())
                logger.info("[mixed-fwd][traj]   A: [A,B@same] vs [A,B@0] (B-step leak into A, want 0): max=%.3e rms=%.3e", d_bstep.max().item(), d_bstep.pow(2).mean().sqrt().item())
            _one([ta_solo])
            _one(grp)
            d = (ta_solo.buffer.latents.float() - ta_mix.buffer.latents.float()).abs()
            logger.info(
                "[mixed-fwd][traj] after step %d (A_solo bs=1, A_mix bs=%d): max=%.3e mean=%.3e",
                s, 1 if s < stag else 2, d.max().item(), d.mean().item(),
            )
        verdict = det_ok and (a_iso_max <= 1e-5) and b_changed_max > 1e-3 and a_self_max > 1e-3
        logger.info("[mixed-fwd] RESULT: %s", "PASS" if verdict else "FAIL")
        logger.info("=" * 72)
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main(load_config_from_cli())
