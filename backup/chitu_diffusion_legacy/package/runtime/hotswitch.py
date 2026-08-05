# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.distributed as dist

from chitu_diffusion.core.distributed.parallel_state import get_cp_group
from chitu_diffusion.observability import Timer
from chitu_diffusion.parallel.state import ContextParallelLatentState, ParallelTaskState
from chitu_diffusion.runtime.parallel_utils import SequencePadder
from chitu_diffusion.runtime.task import DiffusionTask


@dataclass(frozen=True)
class HotswitchDecision:
    action: str
    source_mode: str
    target_mode: str
    step_index: int
    reason: str = ""


@dataclass
class HotswitchResult:
    action: str
    elapsed_ms: float
    migrated_latents: bool
    source_mode: str
    target_mode: str
    step_index: int


class HotswitchStateMigrator:
    """Step-boundary CP/DP layout migration primitives.

    This class prepares task-local state for a scheduler decision. It does not
    rebuild process groups; static execution-group routing owns that concern.
    """

    def prepare(self, task: DiffusionTask, decision: HotswitchDecision) -> HotswitchResult:
        start = time.perf_counter()
        migrated_latents = False
        if decision.action == "cp_to_dp":
            migrated_latents = self._materialize_full_latents(task)
        elif decision.action == "dp_to_cp":
            migrated_latents = self._shard_full_latents(task)
        else:
            raise ValueError(f"Unsupported hotswitch action: {decision.action}")

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        result = HotswitchResult(
            action=decision.action,
            elapsed_ms=elapsed_ms,
            migrated_latents=migrated_latents,
            source_mode=decision.source_mode,
            target_mode=decision.target_mode,
            step_index=decision.step_index,
        )
        task.scheduler_metadata["hotswitch_prepared"] = asdict(result)
        Timer.record_event(
            "hotswitch_prepare",
            {
                "task_id": task.task_id,
                **asdict(result),
            },
        )
        return result

    def _materialize_full_latents(self, task: DiffusionTask) -> bool:
        cp_state = self._cp_latent_state(task)
        if cp_state is None or not cp_state.is_local or task.buffer.latents is None:
            return False
        group = get_cp_group()
        if not dist.is_initialized() or group.group_size <= 1:
            cp_state.is_local = False
            return False
        local_latents = task.buffer.latents.contiguous()
        pieces = [torch.empty_like(local_latents) for _ in range(group.group_size)]
        dist.all_gather(pieces, local_latents, group=group.gpu_group)
        task.buffer.latents = SequencePadder.remove_sequence_padding_and_concat(
            pieces,
            gather_dim=1,
            name="hotswitch_cp_latents",
        )
        cp_state.is_local = False
        if hasattr(task.buffer, "_qwen_cp_latents_local"):
            task.buffer._qwen_cp_latents_local = False
        return True

    def _shard_full_latents(self, task: DiffusionTask) -> bool:
        if task.buffer.latents is None:
            return False
        group = get_cp_group()
        if not dist.is_initialized() or group.group_size <= 1:
            return False
        full_latents = task.buffer.latents
        pieces = SequencePadder.split_sequence_padding(
            full_latents,
            split_num=group.group_size,
            split_dim=1,
            name="hotswitch_cp_latents",
        )
        task.buffer.latents = pieces[group.rank_in_group].contiguous()
        if getattr(task.buffer, "parallel", None) is None:
            task.buffer.parallel = ParallelTaskState()
        task.buffer.parallel.cp_latents = ContextParallelLatentState(
            image_offset=0,
            image_seq_len=int(full_latents.shape[1]),
            is_local=True,
        )
        if hasattr(task.buffer, "_qwen_cp_latents_local"):
            task.buffer._qwen_cp_latents_local = True
        return True

    @staticmethod
    def _cp_latent_state(task: DiffusionTask) -> Optional[ContextParallelLatentState]:
        return getattr(getattr(task.buffer, "parallel", None), "cp_latents", None)
