import numpy as np
import imageio.v2 as imageio


def load_video_frames(video_path: str, max_frames: int = -1) -> np.ndarray:
    frames = []
    reader = imageio.get_reader(video_path)
    try:
        for idx, frame in enumerate(reader):
            if max_frames > 0 and idx >= max_frames:
                break
            if frame.ndim == 2:
                frame = np.stack([frame, frame, frame], axis=-1)
            if frame.shape[-1] == 4:
                frame = frame[..., :3]
            frames.append(frame.astype(np.float32))
    finally:
        reader.close()

    if not frames:
        return np.zeros((0, 0, 0, 3), dtype=np.float32)
    return np.stack(frames, axis=0)


def align_video_pair(gen_frames: np.ndarray, ref_frames: np.ndarray, max_frames: int = 16):
    if len(gen_frames) == 0 or len(ref_frames) == 0:
        return gen_frames, ref_frames

    n = min(len(gen_frames), len(ref_frames))
    if max_frames > 0:
        n = min(n, max_frames)

    gen_idx = np.linspace(0, len(gen_frames) - 1, num=n, dtype=np.int64)
    ref_idx = np.linspace(0, len(ref_frames) - 1, num=n, dtype=np.int64)
    return gen_frames[gen_idx], ref_frames[ref_idx]
