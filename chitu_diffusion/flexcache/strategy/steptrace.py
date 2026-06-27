from __future__ import annotations

import json
import math
import os
from logging import getLogger
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist

from chitu_diffusion.flexcache.flexcache_manager import FlexCacheStrategy
from chitu_diffusion.flexcache.freecache_core import compute_jvp, jvp_predict_noise_pred
from chitu_diffusion.runtime.backend import DiffusionBackend

logger = getLogger(__name__)


class StepTraceStrategy(FlexCacheStrategy):
    """Full-compute step-level trace for MeanCache/FreeCache target analysis."""

    def __init__(self, task, jvp_order: int = 1, save_vectors: bool = False):
        super().__init__()
        self.type = "steptrace"
        self.task_id = str(task.task_id)
        self.total_steps = int(task.req.params.num_inference_steps)
        self.jvp_order = max(0, int(jvp_order))
        self.save_vectors = bool(save_vectors)
        self.tradeoff_score = 0.0
        self.log_dict: Dict[str, List[Any]] = {
            "v": [],
            "latents_pre": [],
            "latents": [],
            "sigmas_pre": [],
            "sigmas": [],
        }
        self.vector_trace: Dict[str, List[Any]] = {
            "guided_v": [],
            "local_v": [],
            "latents_pre": [],
            "latents": [],
            "sigmas_pre": [],
            "sigmas": [],
            "steps": [],
        }
        self.step_trace: List[Dict[str, Any]] = []

    def get_reuse_key(self, **kwargs) -> Optional[str]:
        return None

    def reuse(self, **kwargs) -> torch.Tensor:
        raise RuntimeError("StepTraceStrategy never reuses cached predictions.")

    def get_store_key(self, **kwargs) -> Optional[str]:
        return "steptrace_noise_pred"

    def store(self, **kwargs) -> None:
        step = int(kwargs["step"])
        noise_pred = kwargs["noise_pred"].detach().to(torch.float32)
        sigmas = self._scheduler_sigmas()

        previous_v = self.log_dict["v"][-1] if self.log_dict["v"] else None
        previous_stats = self._pair_stats(noise_pred, previous_v, prefix="prev") if previous_v is not None else {}

        zero_pred_stats = {}
        jvp_pred_stats = {}
        jvp_correction = {}
        if self.log_dict["v"] and sigmas is not None and step + 1 < len(sigmas):
            v_hat0 = jvp_predict_noise_pred(self.log_dict, sigmas, step, order=0).to(torch.float32)
            zero_pred_stats = self._pair_stats(noise_pred, v_hat0, prefix="zero_pred")

            v_hat = jvp_predict_noise_pred(self.log_dict, sigmas, step, order=self.jvp_order).to(torch.float32)
            jvp_pred_stats = self._pair_stats(noise_pred, v_hat, prefix=f"jvp{self.jvp_order}_pred")

            jvp = compute_jvp(self.log_dict, self.jvp_order)
            if jvp is not None:
                delta_sigma = sigmas[step + 1].to(device=jvp.device, dtype=jvp.dtype) - sigmas[step].to(
                    device=jvp.device,
                    dtype=jvp.dtype,
                )
                correction = (jvp.to(torch.float32) * delta_sigma.to(device=jvp.device, dtype=torch.float32)).to(
                    torch.float32
                )
                jvp_correction = self._tensor_stats(correction, prefix=f"jvp{self.jvp_order}_correction")

        row = {
            "step": step,
            "rank": self._rank(),
            "world_size": self._world_size(),
            "cfg_rank": self._cfg_rank(),
            "cp_rank": self._cp_rank(),
            "jvp_order": self.jvp_order,
            "sigma_pre": self._scalar(kwargs["sigma_pre"]),
            "sigma": self._scalar(kwargs["sigma"]),
            **self._tensor_stats(noise_pred, prefix="v"),
            **previous_stats,
            **zero_pred_stats,
            **jvp_pred_stats,
            **jvp_correction,
        }
        self.step_trace.append(row)

        self.log_dict["latents_pre"].append(kwargs["latents_pre"].detach().to(torch.float32))
        self.log_dict["latents"].append(kwargs["latents"].detach().to(torch.float32))
        self.log_dict["sigmas_pre"].append(kwargs["sigma_pre"].detach().to(torch.float32))
        self.log_dict["sigmas"].append(kwargs["sigma"].detach().to(torch.float32))
        self.log_dict["v"].append(noise_pred)
        if self.save_vectors:
            guided_noise_pred = kwargs.get("guided_noise_pred", kwargs["noise_pred"])
            self.vector_trace["guided_v"].append(guided_noise_pred.detach().to(torch.float16).cpu())
            self.vector_trace["local_v"].append(kwargs["noise_pred"].detach().to(torch.float16).cpu())
            self.vector_trace["latents_pre"].append(kwargs["latents_pre"].detach().to(torch.float16).cpu())
            self.vector_trace["latents"].append(kwargs["latents"].detach().to(torch.float16).cpu())
            self.vector_trace["sigmas_pre"].append(kwargs["sigma_pre"].detach().to(torch.float32).cpu())
            self.vector_trace["sigmas"].append(kwargs["sigma"].detach().to(torch.float32).cpu())
            self.vector_trace["steps"].append(step)
        return None

    def wrap_module_with_strategy(self, module: torch.nn.Module) -> None:
        logger.info(
            "StepTrace strategy enabled: jvp_order=%d save_vectors=%s total_steps=%d",
            self.jvp_order,
            self.save_vectors,
            self.total_steps,
        )

    def unwrap_module(self, module: torch.nn.Module) -> None:
        run_output_dir = os.environ.get("CHITU_CURRENT_OUTPUT_DIR", "").strip()
        if run_output_dir and self._should_save_trace():
            self._save_step_trace(run_output_dir)
        logger.info("StepTrace strategy disabled.")

    def reset_state(self) -> None:
        for values in self.log_dict.values():
            values.clear()
        for values in self.vector_trace.values():
            values.clear()
        self.step_trace.clear()
        DiffusionBackend.flexcache.clear_cache()

    def _scheduler_sigmas(self) -> Optional[torch.Tensor]:
        task = DiffusionBackend.generator.current_task
        sampler = getattr(task.buffer, "sampler", None)
        if sampler is not None:
            sigmas = getattr(sampler, "sigmas", None)
            if sigmas is not None:
                return sigmas
        return None

    def _save_step_trace(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        safe_task_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in self.task_id)
        rank_suffix = f"rank{self._rank()}"
        path = os.path.join(output_dir, f"flexcache_steptrace_{safe_task_id}_{rank_suffix}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.step_trace, handle, indent=2)
        logger.info("Saved StepTrace metrics to %s", path)
        if self.save_vectors and self.vector_trace["steps"]:
            vector_path = os.path.join(output_dir, f"flexcache_steptrace_vectors_{safe_task_id}_{rank_suffix}.pt")
            payload = {
                key: torch.stack(values) if values and isinstance(values[0], torch.Tensor) else list(values)
                for key, values in self.vector_trace.items()
            }
            payload["rank"] = self._rank()
            payload["world_size"] = self._world_size()
            payload["cfg_rank"] = self._cfg_rank()
            payload["cp_rank"] = self._cp_rank()
            payload["guidance_scale"] = self._guidance_scale()
            torch.save(payload, vector_path)
            logger.info("Saved StepTrace vectors to %s", vector_path)

    def _should_save_trace(self) -> bool:
        return self._rank() == 0

    def _pair_stats(self, current: torch.Tensor, reference: torch.Tensor, *, prefix: str) -> Dict[str, float]:
        current = current.detach().to(torch.float32)
        reference = reference.detach().to(device=current.device, dtype=torch.float32)
        diff = current - reference
        mse = self._mean_all(diff.square())
        rmse = math.sqrt(max(mse, 0.0))
        ref_mse = self._mean_all(current.square())
        rel_mse = mse / max(ref_mse, 1e-20)
        rel_rmse = math.sqrt(max(rel_mse, 0.0))
        dot = self._sum_all(current * reference)
        current_norm = math.sqrt(max(self._sum_all(current.square()), 0.0))
        reference_norm = math.sqrt(max(self._sum_all(reference.square()), 0.0))
        cosine = dot / max(current_norm * reference_norm, 1e-20)
        return {
            f"{prefix}_mse": mse,
            f"{prefix}_rmse": rmse,
            f"{prefix}_rel_mse": rel_mse,
            f"{prefix}_rel_rmse": rel_rmse,
            f"{prefix}_cosine": cosine,
        }

    def _tensor_stats(self, tensor: torch.Tensor, *, prefix: str) -> Dict[str, float]:
        tensor = tensor.detach().to(torch.float32)
        return {
            f"{prefix}_mean": self._mean_all(tensor),
            f"{prefix}_mse_to_zero": self._mean_all(tensor.square()),
            f"{prefix}_norm": math.sqrt(max(self._sum_all(tensor.square()), 0.0)),
        }

    def _mean_all(self, tensor: torch.Tensor) -> float:
        local = torch.stack(
            [
                tensor.detach().to(torch.float64).sum(),
                torch.tensor(float(tensor.numel()), device=tensor.device, dtype=torch.float64),
            ]
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local, op=dist.ReduceOp.SUM)
        return float((local[0] / local[1].clamp_min(1.0)).item())

    def _sum_all(self, tensor: torch.Tensor) -> float:
        value = tensor.detach().to(torch.float64).sum()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return float(value.item())

    def _scalar(self, value: torch.Tensor) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().to(torch.float32).mean().item())
        return float(value)

    def _rank(self) -> int:
        return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    def _world_size(self) -> int:
        return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

    def _cfg_rank(self) -> int:
        try:
            from chitu_diffusion.core.distributed.parallel_state import get_cfg_group

            return int(get_cfg_group().rank_in_group)
        except Exception:
            return 0

    def _cp_rank(self) -> int:
        try:
            from chitu_diffusion.core.distributed.parallel_state import get_cp_group

            return int(get_cp_group().rank_in_group)
        except Exception:
            return 0

    def _guidance_scale(self) -> Optional[float]:
        try:
            task = DiffusionBackend.generator.current_task
            args = getattr(getattr(DiffusionBackend, "generator", None), "args", None)
            if args is not None:
                return float(args.models.sampler.guidance_scale[0])
            return float(task.req.params.guidance_scale)
        except Exception:
            return None
