#!/usr/bin/env python3
"""Compatibility entry point for the Full-Mesh CuTe installer."""

import runpy
from pathlib import Path

TARGET = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "install"
    / "install_full_mesh_cute.py"
)
runpy.run_path(str(TARGET), run_name="__main__")
