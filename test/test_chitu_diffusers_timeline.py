from pathlib import Path

import pytest

from chitu_diffusers.benchmarks.plot_strategy_timeline import (
    Run,
    derive_deadlines_ms,
    parse_fixed_lane_segments,
    parse_rank_timeline,
)
from chitu_diffusers.epac.timeline import RankTimelineRecorder


def _run_with_cp1_cost(name: str, *, image_tokens: int, step_ms: float) -> Run:
    return Run(
        name=name,
        phases=[],
        segments=[],
        schedule_summary={},
        metrics={"per_request": {}},
        step_cost_ms={(image_tokens, 1): step_ms},
    )


def test_default_deadline_uses_median_cp1_cost_and_explicit_slo_wins() -> None:
    requests = [
        {
            "request_id": "default",
            "arrival_ms": 500.0,
            "width": 1024,
            "height": 1024,
            "num_steps": 10,
        },
        {
            "request_id": "explicit",
            "arrival_ms": 700.0,
            "width": 1024,
            "height": 1024,
            "num_steps": 10,
            "deadline_ms": 900.0,
        },
    ]
    runs = [
        _run_with_cp1_cost("a", image_tokens=4096, step_ms=100.0),
        _run_with_cp1_cost("b", image_tokens=4096, step_ms=120.0),
    ]

    deadlines, sources = derive_deadlines_ms(requests, runs, default_factor=1.2)

    assert deadlines["default"] == pytest.approx(500.0 + 1.2 * 110.0 * 10)
    assert sources["default"] == "cp1_default"
    assert deadlines["explicit"] == pytest.approx(1600.0)
    assert sources["explicit"] == "explicit"


def test_default_deadline_requires_cp1_warmup_measurement() -> None:
    request = {
        "request_id": "missing",
        "arrival_ms": 0.0,
        "width": 1024,
        "height": 1024,
        "num_steps": 10,
    }

    with pytest.raises(ValueError, match="cp1 warmup"):
        derive_deadlines_ms([request], [], default_factor=1.2)


def test_fixed_cp_timeline_classifies_profiled_finalize(tmp_path: Path) -> None:
    log_path = tmp_path / "service.log"
    log_path.write_text(
        "static_cp lane complete: request=req0000 ranks=[0, 1] "
        "start_unix_ms=1000.000 denoise_start_unix_ms=1100.000 "
        "end_unix_ms=1900.000 steps=2 predicted=100.00ms/step "
        "measured=100.00ms/step finalize=500.00ms vae=200.00ms "
        "postprocess=100.00ms png=150.00ms png_bytes=1024 "
        "cuda_alloc_before=10.0MiB cuda_peak=20.0MiB "
        "latent_contiguous=True\n",
        encoding="utf-8",
    )

    parsed = parse_fixed_lane_segments(log_path, world_size=2)

    assert parsed is not None
    segments, summary = parsed
    assert summary["prepare_gpu_ms"] == pytest.approx(200.0)
    assert summary["vae_gpu_ms"] == pytest.approx(200.0)
    assert summary["postprocess_gpu_ms"] == pytest.approx(100.0)
    assert summary["png_gpu_ms"] == pytest.approx(150.0)
    assert summary["lane_wait_gpu_ms"] == pytest.approx(600.0)
    assert summary["outside_dit_gpu_ms"] == pytest.approx(150.0)
    assert {
        (segment.gpu, segment.kind)
        for segment in segments
        if segment.kind != "outside_dit"
    } >= {
        (0, "prepare"),
        (0, "compute"),
        (0, "vae"),
        (0, "postprocess"),
        (0, "png"),
        (1, "lane_wait"),
    }


def test_rank_timeline_recorder_flushes_and_parser_uses_absolute_spans(
    tmp_path: Path,
) -> None:
    rank0 = RankTimelineRecorder(rank=0, output_root=tmp_path, enabled=True)
    rank1 = RankTimelineRecorder(rank=1, output_root=tmp_path, enabled=True)
    rank0.instant(
        "request_submit",
        unix_ns=1_000_000_000,
        request_id="req0000",
    )
    for recorder in (rank0, rank1):
        recorder.record(
            "denoise",
            start_unix_ns=1_100_000_000,
            end_unix_ns=1_300_000_000,
            request_id="req0000",
            lane_ranks=(0, 1),
            metadata={"epoch": 2},
        )
    rank0.record(
        "vae",
        start_unix_ns=1_300_000_000,
        end_unix_ns=1_350_000_000,
        request_id="req0000",
        lane_ranks=(0, 1),
    )
    rank0.record(
        "cpu_png",
        start_unix_ns=1_350_000_000,
        end_unix_ns=1_375_000_000,
        request_id="req0000",
        lane_ranks=(0, 1),
        resource="cpu",
    )
    rank0.close()
    rank1.close()

    parsed = parse_rank_timeline(tmp_path, world_size=2)

    assert parsed is not None
    segments, summary = parsed
    assert summary["timeline_source"] == "rank_events"
    assert summary["compute_gpu_ms"] == pytest.approx(400.0)
    assert summary["vae_gpu_ms"] == pytest.approx(50.0)
    assert summary["cpu_png_ms"] == pytest.approx(25.0)
    assert any(
        segment.gpu == 0
        and segment.kind == "compute"
        and segment.start_ms == pytest.approx(100.0)
        for segment in segments
    )
    assert any(segment.gpu == 2 and segment.kind == "cpu_png" for segment in segments)


def test_disabled_rank_timeline_recorder_writes_nothing(tmp_path: Path) -> None:
    recorder = RankTimelineRecorder(rank=0, output_root=tmp_path, enabled=False)
    recorder.record("denoise", start_unix_ns=1, end_unix_ns=2)
    recorder.close()

    assert not list(tmp_path.iterdir())
