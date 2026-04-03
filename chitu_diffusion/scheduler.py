# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from logging import getLogger
import time
from typing import List

from chitu_diffusion.task import DiffusionTaskPool, DiffusionTaskStatus

logger = getLogger(__name__)


class DiffusionScheduler:
    '''
    Naive diffusion scheduler, which schedules tasks in a FIFO manner, one task at a time.
    '''
    
    @staticmethod
    def build(infer_args):
        logger.info(f"Building diffusion scheduler: {infer_args=}")
        return DiffusionScheduler(infer_args)
    
    def __init__(self, args):
        self.scheduling_ts = 0
        logger.info(f"Initialized FIFO DiffusionScheduler.")

    def schedule(self) -> List[str]:
        """
        First in first out, naive scheduling
        
        Returns:
            List[str]: contains only one task_id
        """
        # 检查任务池是否为空
        if DiffusionTaskPool.is_empty():
            logger.info("DiffusionTaskPool is empty, returning empty task list.")
            return []
        
        # 更新调度时间戳
        self.scheduling_ts = time.perf_counter_ns()
        
        # 获取所有可调度的任务（状态为Pending的任务）
        available_task_ids = [
            task_id for task_id in DiffusionTaskPool.id_list
            if DiffusionTaskPool.pool[task_id].status == DiffusionTaskStatus.Pending
        ]
        
        # 如果没有可调度的任务
        if not available_task_ids:
            logger.info("No pending tasks available for scheduling.")
            return []
        
        # FIFO调度：选择队列中的第一个任务
        selected_task_id = available_task_ids[0]
        selected_task = DiffusionTaskPool.pool[selected_task_id]
        
        # 更新任务的调度时间戳
        selected_task.sched_ts = self.scheduling_ts
        
        logger.debug(f"Scheduled task: {selected_task_id} (type: {selected_task.task_type}")
        
        return [selected_task_id]

    def can_schedule(self) -> bool:
        """
        检查是否可以进行调度
        
        Returns:
            bool: 如果有待调度的任务则返回True
        """
        if DiffusionTaskPool.is_empty():
            return False
        
        # 检查是否有Pending状态的任务
        for task_id in DiffusionTaskPool.id_list:
            task = DiffusionTaskPool.pool[task_id]
            if task.status == DiffusionTaskStatus.Pending:
                return True
        
        return False

    def get_queue_length(self) -> int:
        """
        获取当前待调度任务数量
        
        Returns:
            int: 待调度任务数量
        """
        if DiffusionTaskPool.is_empty():
            return 0
        
        pending_count = sum(
            1 for task_id in DiffusionTaskPool.id_list
            if DiffusionTaskPool.pool[task_id].status == DiffusionTaskStatus.Pending
        )
        
        return pending_count

    def get_scheduler_info(self) -> dict:
        """
        获取调度器信息
        
        Returns:
            dict: 包含调度器状态信息的字典
        """
        return {
            "scheduler_type": self.scheduler_type,
            "total_tasks": len(DiffusionTaskPool.pool),
            "pending_tasks": self.get_queue_length(),
            "can_schedule": self.can_schedule(),
            "last_scheduling_ts": self.scheduling_ts
        }
        
