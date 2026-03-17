import os
from pathlib import Path
from logging import getLogger
from typing import Dict, Tuple
from chitu_diffusion.task import DiffusionTaskPool, DiffusionTaskStatus
from chitu_diffusion.utils.output_naming import build_video_name_from_task
logger = getLogger(__name__)


def collect_videos_and_prompts(args) -> Tuple[Dict[str, str], str]:
    f"""
    从 DiffusionTaskPool 收集已完成任务的视频
    Returns:
        video_prompt: dict, key 为带后缀视频文件名，value 是对应 prompt
        videos_dir: str, 视频所在目录（绝对路径）
    """
    completed_tasks = _get_completed_tasks_from_pool()
    if not completed_tasks:
        raise ValueError("No completed tasks found in DiffusionTaskPool")

    logger.info(f"Found {len(completed_tasks)} completed tasks")

    save_dirs = {str(Path(task.req.params.save_dir).resolve()) for task in completed_tasks}
    if len(save_dirs) != 1:
        raise ValueError(
            f"Completed tasks have different save_dir values: {sorted(save_dirs)}. "
            "VBench expects a single videos_dir. Please unify save_dir or copy videos into one folder."
        )
    videos_dir = next(iter(save_dirs))

    video_prompt: Dict[str, str] = {}
    missing_files = 0

    for task in completed_tasks:
        prompt = task.req.get_prompt()
        save_dir = str(Path(task.req.params.save_dir).resolve())
        save_name = build_video_name_from_task(task)
        video_path = os.path.join(save_dir, save_name)

        if not os.path.exists(video_path):
            missing_files += 1
            logger.warning(f"Video file not found: {video_path}")
            continue

        if save_name in video_prompt:
            logger.warning(f"Duplicate video name detected: {save_name}. Overwriting previous prompt.")
        video_prompt[save_name] = prompt

        logger.debug(f"Collected: {save_name} -> {prompt[:50]}...")

    if not video_prompt:
        raise ValueError("No video files found for completed tasks (all missing on disk?)")

    logger.info(
        f"Successfully collected {len(video_prompt)} videos. "
        f"Missing files: {missing_files}. videos_dir={videos_dir}"
    )
    return video_prompt, videos_dir


def _get_completed_tasks_from_pool():
    """
    从 DiffusionTaskPool 获取所有已完成的任务
    """


    completed_tasks = []
    for task_id, task in DiffusionTaskPool.pool.items():
        if task.status == DiffusionTaskStatus.Completed:
            completed_tasks.append(task)
        else:
            logger.debug(f"Skipping task {task_id} with status {task.status}")

    return completed_tasks
