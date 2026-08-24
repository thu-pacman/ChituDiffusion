from __future__ import annotations

import json
import shutil
import subprocess

import pytest
import torch
from torch import nn

from chitu_diffusion.models.minimax_h3.audio_vae import MiniMaxH3AudioVAE
from chitu_diffusion.models.minimax_h3.media_export import export_mp4
from chitu_diffusion.models.minimax_h3.video_vae import MiniMaxH3VideoVAE


class _FakeVideoProcessor:
    def revert_tensor(self, decoded: torch.Tensor) -> torch.Tensor:
        return decoded


class _FakeVideoModel(nn.Module):
    decoder_tiling = False

    def __init__(self) -> None:
        super().__init__()
        self.processor = _FakeVideoProcessor()
        self.seen: torch.Tensor | None = None
        self.seen_tiling: bool | None = None

    def decode_base(self, latents: torch.Tensor) -> torch.Tensor:
        self.seen = latents.clone()
        self.seen_tiling = self.decoder_tiling
        batch, _, frames, height, width = latents.shape
        return torch.full(
            (batch, 3, frames, height, width),
            0.25,
            device=latents.device,
            dtype=latents.dtype,
        )

    def tiled_decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decode_base(latents)


class _FakeAudioModel(nn.Module):
    sample_rate = 32000

    def __init__(self) -> None:
        super().__init__()
        self.seen: torch.Tensor | None = None

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        self.seen = latents.clone()
        return latents[:, :1, :]


def test_video_vae_restores_channel_stats_and_canonicalizes_output() -> None:
    model = _FakeVideoModel()
    wrapper = MiniMaxH3VideoVAE(
        model,
        latents_mean=list(range(24)),
        latents_std=[2.0] * 24,
        require_tiled_decoder=True,
    )
    output = wrapper.decode(torch.ones(1, 24, 2, 4, 6))

    assert model.decoder_tiling is False
    assert model.seen_tiling is False
    assert model.seen is not None
    torch.testing.assert_close(
        model.seen[0, :, 0, 0, 0],
        torch.arange(24, dtype=torch.float32) + 2,
    )
    assert output.shape == (1, 3, 2, 4, 6)
    assert output.dtype == torch.float32
    assert output.is_contiguous()
    assert output.min() >= 0 and output.max() <= 1


def test_video_vae_fails_closed_without_release_interfaces() -> None:
    with pytest.raises(TypeError, match="decode_base"):
        MiniMaxH3VideoVAE(nn.Identity(), [0.0] * 24, [1.0] * 24)


def test_video_vae_enables_tiling_for_full_temporal_clip() -> None:
    model = _FakeVideoModel()
    wrapper = MiniMaxH3VideoVAE(
        model,
        latents_mean=[0.0] * 24,
        latents_std=[1.0] * 24,
        require_tiled_decoder=True,
    )
    wrapper.decode(torch.zeros(1, 24, 7, 2, 2))
    assert model.seen_tiling is True
    assert model.decoder_tiling is False


def test_audio_vae_restores_stats_and_returns_stereo_batch() -> None:
    model = _FakeAudioModel()
    wrapper = MiniMaxH3AudioVAE(
        model,
        latents_mean=list(range(32)),
        latents_std=[3.0] * 32,
    )
    output = wrapper.decode(torch.ones(2, 32, 5))

    assert model.seen is not None
    torch.testing.assert_close(
        model.seen[0, :, 0],
        torch.arange(32, dtype=torch.float32) + 3,
    )
    assert output.shape == (1, 2, 5)
    assert output.dtype == torch.float32
    assert wrapper.sample_rate == 32000


def test_invalid_latent_stats_fail_closed() -> None:
    with pytest.raises(ValueError, match="latents_std"):
        MiniMaxH3AudioVAE(_FakeAudioModel(), [0.0] * 32, [1.0] * 31 + [0.0])


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="system ffmpeg/ffprobe are required for the real mux contract test",
)
def test_export_mp4_muxes_h264_yuv420p_and_stereo_aac(tmp_path) -> None:
    video = torch.zeros(1, 3, 2, 16, 16)
    video[:, 0, 1] = 1
    audio = torch.zeros(1, 2, 3200)
    payload = export_mp4(video, audio, fps=24, sample_rate=32000)
    output = tmp_path / "output.mp4"
    output.write_bytes(payload)

    probe = subprocess.run(
        [
            shutil.which("ffprobe") or "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name,pix_fmt,sample_rate,channels",
            "-of",
            "json",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(probe.stdout)["streams"]
    video_stream = next(item for item in streams if item["codec_type"] == "video")
    audio_stream = next(item for item in streams if item["codec_type"] == "audio")
    assert video_stream["codec_name"] == "h264"
    assert video_stream["pix_fmt"] == "yuv420p"
    assert audio_stream["codec_name"] == "aac"
    assert audio_stream["sample_rate"] == "32000"
    assert audio_stream["channels"] == 2
