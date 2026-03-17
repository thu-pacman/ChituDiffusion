# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import time
from functools import wraps
from typing import Any, Dict
from contextvars import ContextVar
from contextlib import contextmanager
from logging.config import dictConfig

try:
    import torch.distributed as dist

    IS_DIST = True
except ImportError:
    IS_DIST = False


_log_context: ContextVar[Dict[str, Any]] = ContextVar("chitu_log_context", default={})

CHITU_LOGGING_LEVEL = os.getenv("CHITU_LOGGING_LEVEL", "INFO")

_FORMAT = (
    f"%(levelname)s %(asctime)s "
    f"%(rank)s [%(name)s:%(lineno)d] %(context)s %(message)s"
)

_DATE_FORMAT = "%m-%d %H:%M:%S"


class StageColor:
    RESET = "\033[0m"
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    GREEN = "\033[32m"


_STAGE_COLOR_MAP = {
    "TextEncode": StageColor.CYAN,
    "VAEEncode": StageColor.YELLOW,
    "Denoise": StageColor.BLUE,
    "VAEDecode": StageColor.GREEN,
}


def _get_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def should_log_info_on_rank() -> bool:
    if not _get_bool_env("CHITU_LOG_RANK0_ONLY", True):
        return True
    if IS_DIST and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def should_emit_progress(step: int, total: int, interval: int) -> bool:
    if total <= 0:
        return False
    safe_interval = max(1, interval)
    return step == total or step == 1 or (step % safe_interval == 0)


def colorize_stage(stage_name: str) -> str:
    if not _get_bool_env("CHITU_ENABLE_STAGE_COLOR", True):
        return stage_name
    color = _STAGE_COLOR_MAP.get(stage_name, "")
    if not color:
        return stage_name
    return f"{color}{stage_name}{StageColor.RESET}"


def _task_prefix(tag: str, task_id: str) -> str:
    return f"[{tag:<8}] task_id={task_id} |"


def log_stage(
    logger: logging.Logger,
    stage_name: str,
    event: str,
    task_id: str,
    extra: str = "",
) -> None:
    if not should_log_info_on_rank():
        return
    colored_stage = colorize_stage(stage_name)
    suffix = f" {extra}" if extra else ""
    logger.info(f"{_task_prefix('STAGE', task_id)} event={event:<5} stage={colored_stage}{suffix}")


def log_progress(
    logger: logging.Logger,
    stage_name: str,
    task_id: str,
    step: int,
    total: int,
    interval: int,
    timestep: Any = None,
) -> None:
    if not should_log_info_on_rank():
        return
    if not should_emit_progress(step=step, total=total, interval=interval):
        return
    percent = (step / total) * 100.0 if total > 0 else 0.0
    detail = f" timestep={timestep}" if timestep is not None else ""
    logger.info(
        f"{_task_prefix('PROGRESS', task_id)} stage={colorize_stage(stage_name)} "
        f"step={step:>3}/{total:<3} pct={percent:>5.1f}%{detail}"
    )


def log_result(logger: logging.Logger, task_id: str, message: str) -> None:
    if not should_log_info_on_rank():
        return
    logger.info(f"{_task_prefix('RESULT', task_id)} {message}")


def log_perf(logger: logging.Logger, task_id: str, stage_name: str, elapsed_ms: float) -> None:
    if not should_log_info_on_rank():
        return
    logger.info(
        f"{_task_prefix('PERF', task_id)} stage={stage_name:<9} elapsed={elapsed_ms:.2f}ms"
    )


class ChituFormatter(logging.Formatter):

    def __init__(self, fmt=None, datefmt=None, style="%"):
        super().__init__(fmt, datefmt, style)

    def format(self, record: logging.LogRecord) -> str:
        original_msg = record.getMessage()

        if IS_DIST and dist.is_initialized():
            rank = dist.get_rank()
            record.rank = f"[Rank {rank}]"
        else:
            record.rank = ""

        context = _log_context.get()
        if context:
            context_str = " ".join([f"{k}={v}" for k, v in context.items()])
            record.context = f"[{context_str}]"
        else:
            record.context = ""

        record.msg = original_msg
        record.args = None
        return super().format(record)


DEFAULT_CHITU_LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "chitu": {
            "class": "chitu_core.logging_utils.ChituFormatter",
            "format": _FORMAT,
            "datefmt": _DATE_FORMAT,
        },
    },
    "handlers": {
        "chitu": {
            "class": "logging.StreamHandler",
            "formatter": "chitu",
            "level": CHITU_LOGGING_LEVEL,
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "chitu": {
            "level": CHITU_LOGGING_LEVEL,
            "handlers": ["chitu"],
            "propagate": False,
        },
    },
    "root": {
        "level": CHITU_LOGGING_LEVEL,
        "handlers": ["chitu"],
    },
}


@contextmanager
def log_context(**kwargs):

    old_context = _log_context.get()
    new_context = old_context.copy()
    new_context.update(kwargs)

    try:
        _log_context.set(new_context)
        yield
    finally:
        _log_context.set(old_context)


def configure_chitu_logging():
    dictConfig(DEFAULT_CHITU_LOGGING_CONFIG)


def setup_chitu_logging():
    configure_chitu_logging()
