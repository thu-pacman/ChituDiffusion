from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_installer():
    path = Path(__file__).resolve().parents[1] / "script" / "install_full_mesh_cute.py"
    spec = importlib.util.spec_from_file_location("install_full_mesh_cute", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_streaming_overlay_upgrade_is_idempotent(tmp_path: Path) -> None:
    installer = _load_installer()
    source = tmp_path / "flash_fwd_sm90.py"
    source.write_text(
        installer._TOKEN_START_STREAMING_METADATA
        + installer._TOKEN_START_WAIT_BLOCK,
        encoding="utf-8",
    )

    assert installer._apply(
        source,
        [],
        False,
        upgrades=installer.SM90_UPGRADES,
    )
    assert not installer._apply(
        source,
        [],
        True,
        upgrades=installer.SM90_UPGRADES,
    )

    updated = source.read_text(encoding="utf-8")
    assert installer._NEW_STREAMING_METADATA in updated
    assert installer.WAIT_BLOCK in updated
    assert "self.tile_n - 1) // self.tile_n" in updated
    assert "block_start = n_block * self.tile_n" not in updated


def test_ready_mask_patch_carries_scheduler_and_stage_metadata() -> None:
    patch = (
        Path(__file__).resolve().parents[1]
        / "script/overlays/full_mesh_cute/ready_mask.patch"
    ).read_text(encoding="utf-8")

    assert "self.streaming_ready_mask" in patch
    assert "def _next_ready_block(" in patch
    assert "sBlockIds" in patch
    assert "cute.arch.shuffle_sync(observed, 0)" in patch

    net_win_patch = (
        Path(__file__).resolve().parents[1]
        / "script/overlays/full_mesh_cute/ready_mask_net_win.patch"
    ).read_text(encoding="utf-8")
    assert "cached_ready_mask" in net_win_patch
    assert "self.intra_wg_overlap = intra_wg_overlap" in net_win_patch
    assert "sBlockIds[kv_producer_state.index] = n_block" in net_win_patch
