from __future__ import annotations

import math
from dataclasses import dataclass

MINIMAX_H3_FPS = 24
MINIMAX_H3_AUDIO_LATENT_HZ = 40
MINIMAX_H3_CANVAS_MULTIPLE = 32
MINIMAX_H3_BASE_SHORT_EDGE = 768
MINIMAX_H3_MAX_PIXELS = 768 * 1344
MINIMAX_H3_VIDEO_SPATIAL_COMPRESSION = 16


def align_frame_count(frame_count: int) -> int:
    """Snap a request up to the H3 ``17n+5`` frame boundary."""

    if isinstance(frame_count, bool):
        raise TypeError("frame_count must be an integer")
    frame_count = int(frame_count)
    if frame_count <= 0:
        return 1
    return frame_count + (5 - frame_count) % 17


def video_latent_t(frame_count: int) -> int:
    """Return the H3 video-VAE temporal size (``5n+2``)."""

    frame_count = int(frame_count)
    if frame_count <= 5:
        return 2
    return ((frame_count - 5) // 17) * 5 + 2


def frame_count_from_video_latent_t(latent_t: int) -> int:
    latent_t = int(latent_t)
    if latent_t == 1:
        return 1
    if latent_t < 2 or (latent_t - 2) % 5:
        raise ValueError("video latent T must be 1 or match 5n+2")
    return 17 * ((latent_t - 2) // 5) + 5


def audio_latent_t(duration_seconds: float) -> int:
    """Round at the native 40 Hz audio-latent boundary."""

    duration_seconds = float(duration_seconds)
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive and finite")
    return round(duration_seconds * MINIMAX_H3_AUDIO_LATENT_HZ)


def _nearest_multiple(value: float, multiple: int) -> int:
    return max(multiple, round(float(value) / multiple) * multiple)


@dataclass(frozen=True, slots=True)
class FL2VACanvas:
    width: int
    height: int
    source_width: float
    source_height: float
    size_mode: str
    multiple: int = MINIMAX_H3_CANVAS_MULTIPLE

    @property
    def latent_width(self) -> int:
        return self.width // MINIMAX_H3_VIDEO_SPATIAL_COMPRESSION

    @property
    def latent_height(self) -> int:
        return self.height // MINIMAX_H3_VIDEO_SPATIAL_COMPRESSION


def resolve_canvas(
    width: float,
    height: float,
    *,
    short_edge: int = MINIMAX_H3_BASE_SHORT_EDGE,
    max_pixels: int = MINIMAX_H3_MAX_PIXELS,
) -> FL2VACanvas:
    """Apply SGLang H3 adaptive canvas math and nearest 32px rounding."""

    source_width = float(width)
    source_height = float(height)
    if (
        not math.isfinite(source_width)
        or not math.isfinite(source_height)
        or source_width <= 0
        or source_height <= 0
    ):
        raise ValueError("canvas width and height must be positive and finite")
    if short_edge <= 0 or max_pixels <= 0:
        raise ValueError("short_edge and max_pixels must be positive")
    ratio = source_width / source_height
    if not 0.25 <= ratio <= 4.0:
        raise ValueError("canvas aspect ratio must be within [1:4, 4:1]")

    if ratio >= 1.0:
        nominal_width, nominal_height = short_edge * ratio, float(short_edge)
    else:
        nominal_width, nominal_height = float(short_edge), short_edge / ratio
    nominal_area = nominal_width * nominal_height
    if nominal_area > max_pixels:
        size_mode = "area"
        scale = math.sqrt(max_pixels / nominal_area)
        nominal_width *= scale
        nominal_height *= scale
    else:
        size_mode = "short_edge"

    return FL2VACanvas(
        width=_nearest_multiple(nominal_width, MINIMAX_H3_CANVAS_MULTIPLE),
        height=_nearest_multiple(nominal_height, MINIMAX_H3_CANVAS_MULTIPLE),
        source_width=source_width,
        source_height=source_height,
        size_mode=size_mode,
    )


@dataclass(frozen=True, slots=True)
class FL2VATimeRequest:
    """Pipeline-facing temporal and display request."""

    duration_seconds: float
    width: float
    height: float
    seed: int = 0
    fps: int = MINIMAX_H3_FPS


@dataclass(frozen=True, slots=True)
class FL2VAResolvedPlan:
    requested_frame_count: int
    frame_count: int
    duration_seconds: float
    fps: int
    video_latent_t: int
    audio_latent_t: int
    canvas: FL2VACanvas
    seed: int

    @property
    def latent_height(self) -> int:
        return self.canvas.latent_height

    @property
    def latent_width(self) -> int:
        return self.canvas.latent_width


def resolve_fl2va_plan(request: FL2VATimeRequest) -> FL2VAResolvedPlan:
    """Resolve one request into exact temporal and spatial latent geometry."""

    if request.fps != MINIMAX_H3_FPS:
        raise ValueError(f"MiniMax H3 fps is fixed at {MINIMAX_H3_FPS}")
    duration = float(request.duration_seconds)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration_seconds must be positive and finite")
    if isinstance(request.seed, bool) or request.seed < 0:
        raise ValueError("seed must be a non-negative integer")
    requested_frames = round(duration * request.fps)
    frames = align_frame_count(requested_frames)
    resolved_duration = frames / request.fps
    return FL2VAResolvedPlan(
        requested_frame_count=requested_frames,
        frame_count=frames,
        duration_seconds=resolved_duration,
        fps=request.fps,
        video_latent_t=video_latent_t(frames),
        audio_latent_t=audio_latent_t(resolved_duration),
        canvas=resolve_canvas(request.width, request.height),
        seed=int(request.seed),
    )


# SGLang-compatible formula names ease direct pipeline adoption.
minimax_h3_align_frame_count = align_frame_count
minimax_h3_video_latent_t = video_latent_t
minimax_h3_frame_count_from_video_latent_t = frame_count_from_video_latent_t
minimax_h3_audio_latent_t = audio_latent_t
minimax_h3_resolve_spatial_shape = resolve_canvas


__all__ = [
    "MINIMAX_H3_AUDIO_LATENT_HZ",
    "MINIMAX_H3_BASE_SHORT_EDGE",
    "MINIMAX_H3_CANVAS_MULTIPLE",
    "MINIMAX_H3_FPS",
    "MINIMAX_H3_MAX_PIXELS",
    "MINIMAX_H3_VIDEO_SPATIAL_COMPRESSION",
    "FL2VACanvas",
    "FL2VAResolvedPlan",
    "FL2VATimeRequest",
    "align_frame_count",
    "audio_latent_t",
    "frame_count_from_video_latent_t",
    "minimax_h3_align_frame_count",
    "minimax_h3_audio_latent_t",
    "minimax_h3_frame_count_from_video_latent_t",
    "minimax_h3_resolve_spatial_shape",
    "minimax_h3_video_latent_t",
    "resolve_canvas",
    "resolve_fl2va_plan",
    "video_latent_t",
]
