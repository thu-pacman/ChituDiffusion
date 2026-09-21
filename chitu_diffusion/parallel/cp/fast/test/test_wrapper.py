"""Public wrapper ownership/fallback contracts, runnable without CUDA."""

import pytest
import torch

from chitu_diffusion.parallel.cp.fast import WanAdapter, init_fast_cp
from chitu_diffusion.parallel.cp.fast.adapters.base import UnsupportedFastCP


def fixture():
    module = torch.nn.Linear(4, 4).eval()
    inputs = {"hidden_states": torch.ones(1, 2, 4)}
    return module, inputs


@torch.inference_mode()
def test_origin_is_explicit_and_outputs_are_owned():
    module, inputs = fixture()
    with init_fast_cp(device="cpu") as runtime:
        # An origin returning scratch must still get normal callable semantics.
        scratch = torch.empty_like(inputs["hidden_states"])

        def origin(hidden_states):
            return scratch.copy_(module(hidden_states))

        region = runtime.wrap_attention(module, adapter=WanAdapter(), origin=origin)
        with pytest.raises(RuntimeError, match="prepare"):
            region(**inputs)
        region.prepare(inputs)
        first = region(**inputs)
        saved = first.clone()
        assert region.metadata["path"] == "origin"
        region(hidden_states=torch.randn_like(inputs["hidden_states"]))
        torch.testing.assert_close(first, saved, atol=0, rtol=0)
        assert runtime.workspace_count == 0
        with pytest.raises(ValueError, match="metadata changed"):
            region(hidden_states=torch.ones(1, 3, 4))
    with pytest.raises(RuntimeError):
        region(**inputs)


@torch.inference_mode()
def test_bound_values_and_explicit_weight_invalidation():
    module, inputs = fixture()
    runtime = init_fast_cp(device="cpu")
    region = runtime.wrap_attention(
        module, adapter=WanAdapter(), origin=lambda hidden_states: module(hidden_states)
    )
    bound = region.bind(inputs=inputs)
    inputs["hidden_states"].mul_(3)
    torch.testing.assert_close(bound.run(), module(inputs["hidden_states"]))
    region.invalidate()
    module.weight.zero_()
    with pytest.raises(RuntimeError, match="closed or invalidated"):
        bound.run()
    region.prepare(inputs)
    torch.testing.assert_close(region(**inputs), module(inputs["hidden_states"]))
    runtime.close()
    runtime.close()


@torch.inference_mode()
def test_no_silent_fallback_and_no_reentrant_workspace_use():
    module, inputs = fixture()
    with init_fast_cp(device="cpu") as runtime:
        region = runtime.wrap_attention(module, adapter=WanAdapter())
        with pytest.raises(UnsupportedFastCP, match="CP2/4/8"):
            region.prepare(inputs)
        region = runtime.wrap_attention(
            module, adapter=WanAdapter(), origin=lambda hidden_states: hidden_states
        )
        region.prepare(inputs)
        with runtime._execution():
            with pytest.raises(RuntimeError, match="already in use"):
                region(**inputs)
        module.train()
        with pytest.raises(RuntimeError, match="eval weights"):
            region(**inputs)


def test_inference_mode_is_required_before_preparation():
    module, inputs = fixture()
    with init_fast_cp(device="cpu") as runtime:
        region = runtime.wrap_attention(
            module, adapter=WanAdapter(), origin=lambda hidden_states: hidden_states
        )
        with pytest.raises(ValueError, match="inference_mode"):
            region.prepare(inputs)


@torch.inference_mode()
def test_collective_metadata_disagreement_is_detected_before_backend(monkeypatch):
    module, inputs = fixture()
    with init_fast_cp(device="cpu") as runtime:
        region = runtime.wrap_attention(
            module, adapter=WanAdapter(), origin=lambda hidden_states: hidden_states
        )

        def disagree(record):
            other = list(record)
            other[0] = ("different shape or adapter",)
            return [record, tuple(other)]

        monkeypatch.setattr(runtime, "_agree", disagree)
        with pytest.raises(ValueError, match="signatures differ"):
            region.prepare(inputs)
        assert runtime._backend is None


def test_transport_import_does_not_load_model_implementations():
    import subprocess
    import sys

    # A clean interpreter protects lazy imports from test collection order.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import chitu_diffusion.parallel.cp.fast as fast; "
            "assert fast.init_fast_cp and fast.FastAgkvTransport; "
            "assert not any(name in sys.modules for name in ("
            "'chitu_diffusion.parallel.cp.fast.adapters.wan',"
            "'chitu_diffusion.parallel.cp.fast.adapters.qwen_image',"
            "'chitu_diffusion.parallel.cp.fast.adapters.zimage')); "
            "assert 'WanAdapter' in dir(fast)",
        ],
        check=True,
    )


@torch.inference_mode()
def test_borrowers_share_workspace_lock_and_cannot_destroy_owner():
    from types import SimpleNamespace

    destroyed = []
    backend = SimpleNamespace(
        _process_group=None,
        _world_size=1,
        pool_bytes=4096,
        destroy=lambda: destroyed.append(True),
    )
    owner = init_fast_cp(device="cpu", backend=backend)
    owner._owns_backend = True
    borrower = init_fast_cp(device="cpu", backend=backend)
    owner._workspaces["shape"] = "slot"
    assert borrower.workspace_count == 1
    with owner._execution():
        with pytest.raises(RuntimeError, match="already in use"):
            with borrower._execution():
                pass
    with pytest.raises(RuntimeError, match="borrowed wrappers"):
        owner.close()
    borrower.close()
    assert owner.workspace_count == 1
    assert not destroyed
    owner.close()
    assert destroyed == [True]
