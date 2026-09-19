"""Flow UniPC schedule compatibility and isolated FPP sampler previews.

The numerical solvers remain Diffusers implementations. Only the initial flow
sigma grid differs from Diffusers: SmartDiffusion exp/fpp@b64ddaa starts at the
FP32 value 1 - 1/1000 and ends its unshifted grid at zero.
"""

from copy import copy

import numpy as np
from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler


class WanFlowUniPCScheduler(UniPCMultistepScheduler):
    def set_timesteps(
        self, num_inference_steps=None, device=None, sigmas=None, mu=None
    ):
        if sigmas is None:
            if not num_inference_steps or num_inference_steps < 1:
                raise ValueError("num_inference_steps must be positive")
            maximum = float(np.float32(1 - 1 / self.config.num_train_timesteps))
            sigmas = np.linspace(maximum, 0, num_inference_steps + 1)[:-1]
        super().set_timesteps(num_inference_steps, device=device, sigmas=sigmas, mu=mu)


def make_wan_scheduler(solver: str, shift: float):
    if solver == "euler":
        return FlowMatchEulerDiscreteScheduler(shift=shift, use_dynamic_shifting=False)
    if solver == "unipc":
        return WanFlowUniPCScheduler(
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            flow_shift=shift,
            solver_order=2,
            predict_x0=True,
            solver_type="bh2",
            lower_order_final=True,
        )
    raise ValueError("Wan sample_solver must be 'euler' or 'unipc'")


def validate_fpp_scheduler(scheduler):
    if type(scheduler) not in (
        FlowMatchEulerDiscreteScheduler,
        UniPCMultistepScheduler,
        WanFlowUniPCScheduler,
    ):
        raise ValueError(
            "FPP previews support deterministic FlowMatch Euler and UniPC schedulers"
        )
    if getattr(scheduler, "solver_p", None) is not None or scheduler.config.get(
        "stochastic_sampling", False
    ):
        raise ValueError(
            "FPP previews require a deterministic scheduler without an external solver"
        )


def fpp_scheduler_step(scheduler, prediction, timestep, latents, *, commit: bool):
    """Preview without advancing history; commit only the final patch of a step.

    These solvers replace tensors rather than modifying history tensors in
    place. Copy their mutable history lists and scalar state, without copying
    large immutable history tensors on every patch.
    """
    target = scheduler
    if not commit:
        target = copy(scheduler)
        for name in ("model_outputs", "timestep_list"):
            value = getattr(scheduler, name, None)
            if value is not None:
                setattr(target, name, list(value))
    return target.step(prediction, timestep, latents, return_dict=False)[0].to(
        latents.dtype
    )
