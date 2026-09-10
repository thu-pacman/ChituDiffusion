from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_CLI_SPEC = importlib.util.spec_from_file_location(
    "_chitu_diffusion_cli",
    _ROOT / "chitu_diffusion" / "cli.py",
)
assert _CLI_SPEC is not None and _CLI_SPEC.loader is not None
cli = importlib.util.module_from_spec(_CLI_SPEC)
_CLI_SPEC.loader.exec_module(cli)


def test_cli_dispatches_generate_and_serve(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        cli,
        "_run_module",
        lambda module, arguments: calls.append((module, list(arguments))),
    )

    cli.main(["generate", "--model", "zimage", "--prompt", "test"])
    cli.main(["generate", "--model", "flux2-klein", "--steps", "4"])
    cli.main(["generate", "--model", "llada-image", "--steps", "20"])
    cli.main(["serve", "--stage-config", "stage.yaml"])

    assert calls == [
        ("chitu_diffusion.commands.generate.zimage", ["--prompt", "test"]),
        ("chitu_diffusion.commands.generate.flux2_klein", ["--steps", "4"]),
        (
            "chitu_diffusion.commands.generate.llada_image",
            ["--steps", "20"],
        ),
        (
            "chitu_diffusion.commands.serve",
            ["--stage-config", "stage.yaml"],
        ),
    ]


def test_cli_does_not_expose_benchmark() -> None:
    with pytest.raises(SystemExit):
        cli.main(["benchmark"])


def test_zimage_cli_writes_generation_time_and_effective_vae_options(
    monkeypatch, tmp_path, capsys
) -> None:
    from chitu_diffusion.commands.generate import zimage

    output_path = tmp_path / "image.png"
    loaded = {}
    closed = []
    image = SimpleNamespace(save=lambda path: path.write_bytes(b"image"))
    pipeline = SimpleNamespace(
        parallel_context=SimpleNamespace(rank=0, world_size=2),
        last_cache_stats=None,
        last_vae_stats={
            "parallel_vae": False,
            "vae_parallel_degree": 1,
            "vae_parallel_halo": None,
            "vae_decode_mode": "leader",
        },
        generate=lambda request: SimpleNamespace(images=[image]),
        close=lambda: closed.append(True),
    )

    def load(path, **kwargs):
        loaded.update(kwargs)
        return pipeline

    monkeypatch.setattr(zimage.ZImagePipeline, "from_pretrained", load)
    ticks = iter([10.0, 12.5])
    monkeypatch.setattr(zimage, "generation_timestamp", lambda: next(ticks))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "zimage",
            "--model-path",
            "unused",
            "--parallel-vae",
            "--vae-parallel-halo",
            "32",
            "--output",
            str(output_path),
        ],
    )
    zimage.main()
    assert loaded["parallel_vae"] is True
    assert loaded["vae_parallel_halo"] == 32
    metadata = json.loads(output_path.with_suffix(".json").read_text())
    assert metadata["elapsed_s"] == metadata["generate_seconds"] == 2.5
    assert metadata["requested_parallel_vae"] is True
    assert metadata["parallel_vae"] is False
    assert metadata["requested_vae_parallel_halo"] == 32
    assert metadata["vae_parallel_halo"] is None
    assert metadata["vae_decode_mode"] == "leader"
    assert "elapsed_s=2.500" in capsys.readouterr().out
    assert closed == [True]


def test_legacy_config_fields_fail_fast() -> None:
    from chitu_diffusion.serve.config import StageServiceConfig

    with pytest.raises(ValueError, match="legacy chitu_diffusion config field"):
        StageServiceConfig.from_mapping(
            {
                "name": "legacy",
                "gpu": [0],
                "infer": {"max_reqs": 1},
            }
        )


def test_active_sources_do_not_reference_retired_package() -> None:
    root = _ROOT
    active_roots = (root / "chitu_diffusion", root / "test", root / "experiments")
    retired_name = "chitu_" + "diffusers"
    backup_imports = ("from " + "backup", "import " + "backup")

    assert not (root / retired_name).exists()
    for active_root in active_roots:
        for path in active_root.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            assert retired_name not in content
            assert all(fragment not in content for fragment in backup_imports)
