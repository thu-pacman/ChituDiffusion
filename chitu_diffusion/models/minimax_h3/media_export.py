from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def _as_numpy(value: Any):
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _resolve_ffmpeg(configured: str | os.PathLike[str] = "ffmpeg") -> str:
    requested = os.fspath(configured).strip()
    if not requested:
        raise ValueError("ffmpeg_path must not be empty")
    executable = shutil.which(requested)
    if executable is not None:
        return executable
    if requested != "ffmpeg":
        raise RuntimeError(f"configured ffmpeg executable is not usable: {requested}")
    try:
        import imageio_ffmpeg

        executable = imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError, OSError) as exc:
        raise RuntimeError(
            "ffmpeg is required to export MiniMax H3 media; install ffmpeg "
            "or imageio-ffmpeg"
        ) from exc
    if not executable or not Path(executable).is_file():
        raise RuntimeError("imageio-ffmpeg did not provide a usable ffmpeg executable")
    return executable


def _prepare_video(video: Any):
    import numpy as np

    array = _as_numpy(video)
    if array.ndim != 5 or array.shape[0] != 1 or array.shape[1] != 3:
        raise ValueError(
            "video must be shaped [1, 3, F, H, W], got "
            f"{tuple(array.shape)}"
        )
    if array.shape[2] < 1:
        raise ValueError("video must contain at least one frame")
    height, width = int(array.shape[3]), int(array.shape[4])
    if height < 2 or width < 2 or height % 2 or width % 2:
        raise ValueError("H.264 yuv420p export requires positive even height and width")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("video samples must be finite numbers")
    frames = np.transpose(array[0], (1, 2, 3, 0))
    frames = np.rint(np.clip(frames, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.ascontiguousarray(frames)


def _prepare_audio(audio: Any):
    import numpy as np

    array = _as_numpy(audio)
    if array.ndim != 3 or tuple(array.shape[:2]) != (1, 2):
        raise ValueError(
            "audio must be shaped [1, 2, L], got "
            f"{tuple(array.shape)}"
        )
    if array.shape[2] < 1:
        raise ValueError("audio must contain at least one sample")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("audio samples must be finite numbers")
    interleaved = np.transpose(array[0], (1, 0))
    return np.ascontiguousarray(np.clip(interleaved, -1.0, 1.0), dtype=np.float32)


def export_mp4(
    video: Any,
    audio: Any,
    *,
    fps: float = 24.0,
    sample_rate: int = 32000,
    crf: int = 18,
    ffmpeg_path: str | os.PathLike[str] = "ffmpeg",
) -> bytes:
    """Encode canonical H3 tensors as H.264/yuv420p + AAC MP4 bytes."""
    import numpy as np

    if not isinstance(fps, (int, float)) or isinstance(fps, bool) or fps <= 0:
        raise ValueError("fps must be positive")
    if sample_rate != 32000:
        raise ValueError("MiniMax H3 audio export sample_rate must be 32000")
    if not isinstance(crf, int) or isinstance(crf, bool) or not 0 <= crf <= 51:
        raise ValueError("crf must be an integer from 0 to 51")

    frames = _prepare_video(video)
    samples = _prepare_audio(audio)
    frame_count, height, width, _ = frames.shape
    duration_seconds = frame_count / float(fps)
    duration = f"{duration_seconds:.12g}"
    target_samples = round(duration_seconds * sample_rate)
    if samples.shape[0] < target_samples:
        samples = np.pad(
            samples,
            ((0, target_samples - samples.shape[0]), (0, 0)),
        )
    elif samples.shape[0] > target_samples:
        samples = np.ascontiguousarray(samples[:target_samples])
    ffmpeg = _resolve_ffmpeg(ffmpeg_path)

    with tempfile.TemporaryDirectory(prefix="chitu-h3-media-") as temporary:
        root = Path(temporary)
        video_path = root / "video.rgb"
        audio_path = root / "audio.f32"
        video_path.write_bytes(frames.tobytes(order="C"))
        audio_path.write_bytes(samples.tobytes(order="C"))
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            f"{float(fps):.12g}",
            "-i",
            os.fspath(video_path),
            "-f",
            "f32le",
            "-ar",
            str(sample_rate),
            "-ac",
            "2",
            "-i",
            os.fspath(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-t",
            duration,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ar",
            str(sample_rate),
            "-ac",
            "2",
            "-movflags",
            "frag_keyframe+empty_moov+default_base_moof",
            "-f",
            "mp4",
            "pipe:1",
        ]
        try:
            encoded = subprocess.run(
                command,
                check=True,
                capture_output=True,
                timeout=120,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"ffmpeg executable disappeared: {ffmpeg}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("ffmpeg timed out while exporting MiniMax H3 MP4") from exc
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg failed to export MiniMax H3 MP4: {detail}") from exc
    if not encoded.stdout:
        raise RuntimeError("ffmpeg returned an empty MiniMax H3 MP4")
    return encoded.stdout


mux_mp4 = export_mp4

__all__ = ["export_mp4", "mux_mp4"]
