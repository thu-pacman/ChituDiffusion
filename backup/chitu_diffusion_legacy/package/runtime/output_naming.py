import re
from pathlib import Path
from typing import Any, Optional, Tuple

PROMPT_VIDEO_LEN = 64


def slugify_prompt(prompt: Optional[str], max_len: int = PROMPT_VIDEO_LEN) -> str:
    text = (prompt or "").strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[\\/:*?\"<>|]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "prompt"
    return text[:max_len]


def build_video_basename(prompt: Optional[str], seed: Any, step: Any) -> str:
    prompt_slug = slugify_prompt(prompt)
    seed_value = "none" if seed is None else str(seed)
    step_value = "none" if step is None else str(step)
    return f"{prompt_slug}_seed{seed_value}_step{step_value}.mp4"


def build_video_name_from_task(task) -> str:
    params = task.req.params
    return build_video_basename(
        prompt=task.req.get_prompt(),
        seed=getattr(params, "seed", None),
        step=getattr(params, "num_inference_steps", None),
    )


def parse_video_name(filename: str) -> Optional[Tuple[str, str, str]]:
    stem = Path(filename).stem
    m = re.match(r"^(?P<prompt>.+)_seed(?P<seed>[^_]+)_step(?P<step>[^_]+)$", stem)
    if not m:
        return None
    return (
        m.group("prompt").lower(),
        m.group("seed"),
        m.group("step"),
    )
