from __future__ import annotations

import importlib.util
from pathlib import Path

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
    cli.main(["serve", "--stage-config", "stage.yaml"])

    assert calls == [
        ("chitu_diffusion.commands.generate.zimage", ["--prompt", "test"]),
        ("chitu_diffusion.commands.generate.flux2_klein", ["--steps", "4"]),
        (
            "chitu_diffusion.commands.serve",
            ["--stage-config", "stage.yaml"],
        ),
    ]


def test_cli_does_not_expose_benchmark() -> None:
    with pytest.raises(SystemExit):
        cli.main(["benchmark"])


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
