from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_cli_help_does_not_import_model_dependencies() -> None:
    result = subprocess.run(
        [sys.executable, "-S", "-m", "chitu_diffusion.cli", "--help"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "usage: chitu" in result.stdout
    assert "[ERROR]" not in result.stdout
    assert result.stderr == ""
