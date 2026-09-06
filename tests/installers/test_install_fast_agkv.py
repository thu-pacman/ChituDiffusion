from __future__ import annotations

import importlib.util
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SOURCES = _PROJECT_ROOT / "chitu_diffusion" / "parallel" / "cp" / "fast" / "csrc"


def _load_installer():
    path = _PROJECT_ROOT / "tools" / "install" / "install_fast_agkv.py"
    spec = importlib.util.spec_from_file_location("install_fast_agkv", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checkout(tmp_path: Path, installer) -> Path:
    """A minimal fast-ulysses tree carrying just the patched anchors."""

    csrc = tmp_path / "fast_ulysses" / "csrc"
    csrc.mkdir(parents=True)
    (csrc / "bindings.cpp").write_text(
        '#include "a2a_config.cuh"\n'
        "at::Tensor all_to_all_single_4d_ce()\n"
        "{\n" + installer.CE_A2A_LAUNCH_OLD + "}\n"
        "TORCH_LIBRARY(fast_ulysses, m)\n"
        "{\n"
        '    m.impl("all_to_all_single_4d_ce", '
        "c10::DispatchKey::CompositeExplicitAutograd, "
        "&ulysses::all_to_all_single_4d_ce);\n"
        "}\n",
        encoding="utf-8",
    )
    (tmp_path / "fast_ulysses" / "comm.py").write_text(
        "".join(old for old, _ in installer.LOCAL_WORLD_REPLACEMENTS),
        encoding="utf-8",
    )
    (tmp_path / "fast_ulysses" / "__init__.py").write_text(
        '"""test fast_ulysses package"""\n',
        encoding="utf-8",
    )
    return csrc


def test_fast_cp_sources_install_idempotently(tmp_path: Path) -> None:
    installer = _load_installer()
    csrc = _checkout(tmp_path, installer)

    assert installer.apply_overlay(tmp_path, check_only=False)
    assert not installer.apply_overlay(tmp_path, check_only=True)
    assert not installer.apply_overlay(tmp_path, check_only=False)

    bindings = (csrc / "bindings.cpp").read_text(encoding="utf-8")
    assert bindings.count('#include "fast_agkv.cuh"') == 1
    assert bindings.count('m.def("all_gather_kv_4d(') == 1
    assert bindings.count('m.def("ring_ce_prepare_kv_4d(') == 1
    assert bindings.count('m.def("ring_ce_forward_kv_4d(') == 1
    assert bindings.count('m.def("ring_ce_wait_ready(') == 1
    assert bindings.count('m.def("ring_ce_publish_consumed(') == 1
    assert bindings.count('m.def("u2fm2_prepare_qkv(') == 1
    # The copy-engine all-to-all now takes a tuned schedule instead of one
    # stream per peer.
    assert installer.CE_A2A_LAUNCH_OLD not in bindings
    assert "tune_ce_schedule(" in bindings

    comm = (tmp_path / "fast_ulysses" / "comm.py").read_text(encoding="utf-8")
    assert "self.peer_runtime_ranks = list(range(self.world_size))" in comm
    assert "dist.broadcast(uid_t, src=self.peer_global_ranks[0], group=pg)" in comm
    init = (tmp_path / "fast_ulysses" / "__init__.py").read_text(encoding="utf-8")
    assert init.count(installer.SUBGROUP_FEATURE_MARKER.strip()) == 1


def test_every_tracked_source_reaches_the_checkout(tmp_path: Path) -> None:
    installer = _load_installer()
    csrc = _checkout(tmp_path, installer)
    installer.apply_overlay(tmp_path, check_only=False)

    tracked = sorted(
        path.name for path in _SOURCES.iterdir() if path.suffix in {".cu", ".cuh"}
    )
    # The transports are built from these, so a new source that the installer
    # forgets to copy would leave the checkout unbuildable.
    assert "ce_schedule.cu" in tracked and "fast_agkv.cu" in tracked
    for name in tracked:
        installed = csrc / name
        assert installed.is_file(), name
        assert installed.read_bytes() == (_SOURCES / name).read_bytes()


def test_a_missing_launch_anchor_is_reported(tmp_path: Path) -> None:
    installer = _load_installer()
    csrc = _checkout(tmp_path, installer)
    bindings = csrc / "bindings.cpp"
    bindings.write_text(
        bindings.read_text(encoding="utf-8").replace(installer.CE_A2A_LAUNCH_OLD, ""),
        encoding="utf-8",
    )

    try:
        installer.apply_overlay(tmp_path, check_only=False)
    except RuntimeError as error:
        assert "CE all-to-all launch anchor" in str(error)
    else:
        raise AssertionError("a checkout without the launch anchor must be rejected")
