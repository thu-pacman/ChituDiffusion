from __future__ import annotations

import pytest
import torch

from chitu_diffusion.models.hunyuan_image3.api import HunyuanImage3Request
from chitu_diffusion.models.hunyuan_image3.cache import HunyuanImage3Cache
from chitu_diffusion.models.hunyuan_image3.executor import denormalize_image
from chitu_diffusion.models.hunyuan_image3.pipeline import (
    _select_cfg_branch,
    flow_match_sigmas,
)


def test_request_defaults_to_a_square_fifty_step_generation() -> None:
    request = HunyuanImage3Request(prompt="a red cube")

    assert (request.height, request.width) == (1024, 1024)
    assert request.num_steps == 50
    assert request.guidance_scale == 5.0
    assert request.image_paths == ()
    assert request.request_id


def test_request_rejects_unguided_generation() -> None:
    # The released tokenizer always emits a conditional and an unconditional
    # half, so an unguided request would leave half the batch unused.
    with pytest.raises(ValueError, match="guidance_scale"):
        HunyuanImage3Request(prompt="a red cube", guidance_scale=1.0)


def test_cfg_branch_selection_keeps_a_single_batch_entry_recursively() -> None:
    value = {
        "tensor": torch.arange(12).reshape(2, 2, 3),
        "rope": (
            torch.arange(8).reshape(2, 4),
            torch.arange(8, 16).reshape(2, 4),
        ),
        "per_batch_list": [torch.tensor([1]), torch.tensor([2])],
        "shared": torch.ones(3, 4),
    }

    selected = _select_cfg_branch(value, 1)

    torch.testing.assert_close(selected["tensor"], value["tensor"][1:2])
    torch.testing.assert_close(selected["rope"][0], value["rope"][0][1:2])
    assert len(selected["per_batch_list"]) == 1
    torch.testing.assert_close(selected["per_batch_list"][0], torch.tensor([2]))
    assert selected["shared"] is value["shared"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("prompt", None, "prompt"),
        ("prompt", "   ", "prompt"),
        ("height", 0, "at least 16 pixels"),
        ("width", 8, "at least 16 pixels"),
        ("num_steps", 0, "num_steps"),
        ("output_type", "mp4", "output_type"),
        ("deadline_ms", 0.0, "deadline_ms"),
        ("extra_inputs", {"recaption": True}, "extra_inputs"),
    ),
)
def test_request_validation_rejects_unsupported_values(
    field: str, value: object, message: str
) -> None:
    fields = {"prompt": "a red cube"}
    fields[field] = value
    with pytest.raises(ValueError, match=message):
        HunyuanImage3Request(**fields)


def test_flow_match_sigmas_matches_the_released_shifted_schedule() -> None:
    sigmas = flow_match_sigmas(4, shift=5.0)
    base = torch.linspace(1.0, 0.0, 5)
    expected = 5.0 * base / (1 + 4.0 * base)

    torch.testing.assert_close(sigmas, expected)
    assert sigmas[0] == 1.0 and sigmas[-1] == 0.0


def test_unshifted_schedule_is_a_plain_linear_ramp() -> None:
    torch.testing.assert_close(
        flow_match_sigmas(3, shift=1.0), torch.linspace(1.0, 0.0, 4)
    )


def test_denormalize_maps_the_released_range_onto_the_unit_interval() -> None:
    image = torch.tensor([[-1.0, 0.0, 1.0, 4.0]])

    torch.testing.assert_close(
        denormalize_image(image), torch.tensor([[0.0, 0.5, 1.0, 1.0]])
    )


def test_cache_allocates_from_the_first_rank_local_key_states() -> None:
    cache = HunyuanImage3Cache(num_layers=2, max_cache_len=6)
    keys = torch.arange(2 * 1 * 3 * 4, dtype=torch.float32).reshape(2, 1, 3, 4)
    positions = torch.tensor([[0, 1, 2], [0, 1, 2]])

    stored_keys, stored_values = cache.update(keys, keys * -1, 0, {
        "cache_position": positions
    })

    assert stored_keys.shape == (2, 1, 6, 4)
    torch.testing.assert_close(stored_keys[:, :, :3], keys)
    torch.testing.assert_close(stored_values[:, :, :3], keys * -1)
    assert stored_keys[:, :, 3:].count_nonzero() == 0
    assert cache.keys[1] is None
    assert cache.byte_size() == 2 * stored_keys.numel() * stored_keys.element_size()


def test_cache_writes_later_steps_at_their_own_positions() -> None:
    cache = HunyuanImage3Cache(num_layers=1, max_cache_len=4)
    prompt = torch.ones(1, 1, 4, 2)
    cache.update(prompt, prompt, 0, {"cache_position": torch.tensor([[0, 1, 2, 3]])})

    step = torch.full((1, 1, 2, 2), 7.0)
    keys, _ = cache.update(step, step, 0, {"cache_position": torch.tensor([[1, 3]])})

    torch.testing.assert_close(
        keys[0, 0, :, 0], torch.tensor([1.0, 7.0, 1.0, 7.0])
    )


def test_cache_requires_positions_and_a_matching_batch() -> None:
    cache = HunyuanImage3Cache(num_layers=1, max_cache_len=2)
    keys = torch.zeros(2, 1, 2, 2)
    with pytest.raises(ValueError, match="cache positions"):
        cache.update(keys, keys, 0, {})
    with pytest.raises(ValueError, match="cache holds"):
        cache.update(keys, keys, 0, {"cache_position": torch.tensor([[0, 1]])})
