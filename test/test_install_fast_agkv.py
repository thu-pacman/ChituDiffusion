from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_installer():
    path = Path(__file__).resolve().parents[1] / "script" / "install_fast_agkv.py"
    spec = importlib.util.spec_from_file_location("install_fast_agkv", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fast_agkv_overlay_is_idempotent(tmp_path: Path) -> None:
    installer = _load_installer()
    csrc = tmp_path / "fast_ulysses" / "csrc"
    csrc.mkdir(parents=True)
    (csrc / "bindings.cpp").write_text(
        '#include "a2a_config.cuh"\n'
        "TORCH_LIBRARY(fast_ulysses, m)\n"
        "{\n"
        '    m.impl("all_to_all_single_4d_ce", '
        "c10::DispatchKey::CompositeExplicitAutograd, "
        "&ulysses::all_to_all_single_4d_ce);\n"
        "}\n",
        encoding="utf-8",
    )
    (csrc / "ulysses_group.cu").write_text(
        "void init_ce()\n"
        "{\n"
        "    if (!ce_ready_) {\n"
        "        ce_.streams.resize(world_size_);\n"
        "        for (int i = 0; i < world_size_; ++i)\n"
        "            ULYSSES_CUDA_CHECK(cudaStreamCreateWithFlags("
        "&ce_.streams[i], cudaStreamNonBlocking));\n"
        "        ce_ready_ = true;\n"
        "    }\n"
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

    assert installer.apply_overlay(tmp_path, check_only=False)
    assert not installer.apply_overlay(tmp_path, check_only=True)
    assert not installer.apply_overlay(tmp_path, check_only=False)
    bindings = (csrc / "bindings.cpp").read_text(encoding="utf-8")
    assert bindings.count('#include "fast_agkv.cuh"') == 1
    assert bindings.count('m.def("all_gather_kv_4d(') == 1
    assert bindings.count('m.def("full_mesh_stream_kv_4d(') == 1
    assert "bool copy_local=True" in bindings
    assert bindings.count('m.def("full_mesh_wait_ready(') == 1
    assert bindings.count('m.def("full_mesh_publish_consumed(') == 1
    assert bindings.count('m.def("ring_ce_prepare_kv_4d(') == 1
    assert bindings.count('m.def("ring_ce_forward_kv_4d(') == 1
    assert bindings.count('m.def("ring_ce_wait_ready(') == 1
    assert bindings.count('m.def("ring_ce_publish_consumed(') == 1
    comm = (tmp_path / "fast_ulysses" / "comm.py").read_text(encoding="utf-8")
    assert "self.peer_runtime_ranks = list(range(self.world_size))" in comm
    assert "dist.broadcast(uid_t, src=self.peer_global_ranks[0], group=pg)" in comm
    init = (tmp_path / "fast_ulysses" / "__init__.py").read_text(encoding="utf-8")
    assert init.count(installer.SUBGROUP_FEATURE_MARKER.strip()) == 1
    assert (csrc / "fast_agkv.cu").is_file()
    assert (csrc / "fast_agkv.cuh").is_file()
    assert (csrc / "symmetric_pool.cu").is_file()
    assert (csrc / "symmetric_pool.cuh").is_file()
