import os
import json
from pathlib import Path
from logging import getLogger
from typing import Dict, Tuple
from chitu_diffusion.runtime.task import DiffusionTaskPool, DiffusionTaskStatus
from chitu_diffusion.runtime.output_naming import build_video_name_from_task
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

    current_results_dir = os.environ.get("CHITU_CURRENT_RESULTS_DIR", "").strip()
    if current_results_dir:
        videos_dir = str(Path(current_results_dir).resolve())
    else:
        save_dirs = {str(Path(task.req.params.save_dir).resolve()) for task in completed_tasks}
        if len(save_dirs) != 1:
            common = Path(os.path.commonpath(sorted(save_dirs)))
            videos_dir = str(common.resolve())
        else:
            videos_dir = next(iter(save_dirs))

    video_prompt: Dict[str, str] = {}
    missing_files = 0

    for task in completed_tasks:
        prompt = task.req.get_prompt()
        save_dir = str(Path(task.req.params.save_dir).resolve())
        save_name = build_video_name_from_task(task)
        video_path = Path(save_dir) / save_name
        if not video_path.exists():
            image_path = _image_output_from_sidecar(Path(save_dir), task.task_id)
            if image_path is not None:
                video_path = image_path
                save_name = video_path.name

        if not video_path.exists():
            missing_files += 1
            logger.warning(f"Video file not found: {video_path}")
            continue

        try:
            video_key = str(video_path.relative_to(Path(videos_dir)))
        except ValueError:
            video_key = save_name
        if video_key in video_prompt:
            logger.warning(f"Duplicate video key detected: {video_key}. Overwriting previous prompt.")
        video_prompt[video_key] = prompt

        logger.debug(f"Collected: {video_key} -> {prompt[:50]}...")

    if not video_prompt:
        raise ValueError("No video files found for completed tasks (all missing on disk?)")

    logger.info(
        f"Successfully collected {len(video_prompt)} videos. "
        f"Missing files: {missing_files}. videos_dir={videos_dir}"
    )
    return video_prompt, videos_dir


def _image_output_from_sidecar(save_dir: Path, task_id: str) -> Path | None:
    for sidecar_path in sorted(save_dir.glob("*.json")):
        try:
            with open(sidecar_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue
        if not isinstance(payload, dict) or payload.get("task_id") != task_id:
            continue
        filename = payload.get("filename")
        if not filename:
            continue
        output_path = save_dir / str(filename)
        if output_path.exists() and output_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            return output_path
    return None


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
