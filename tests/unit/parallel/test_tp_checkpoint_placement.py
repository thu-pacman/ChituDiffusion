from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from chitu_diffusion.parallel.tp import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
    TensorParallelTopology,
    load_tensor_parallel_checkpoint,
    use_tensor_parallel_topology,
)


class _ExpertBlock(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.gate_and_up_proj = MergedColumnParallelLinear(
            hidden, (intermediate, intermediate), bias=False
        )
        self.down_proj = RowParallelLinear(intermediate, hidden, bias=False)


class _Layer(nn.Module):
    def __init__(self, hidden: int, intermediate: int, owned_experts: list[int]) -> None:
        super().__init__()
        self.qkv_proj = ColumnParallelLinear(hidden, 3 * hidden, bias=False)
        self.experts = nn.ModuleDict(
            {
                str(index): _ExpertBlock(hidden, intermediate)
                for index in owned_experts
            }
        )


def _write_checkpoint(
    directory: Path,
    *,
    hidden: int,
    intermediate: int,
    num_experts: int,
) -> dict[str, torch.Tensor]:
    tensors = {
        "qkv_proj.weight": torch.arange(
            3 * hidden * hidden, dtype=torch.float32
        ).reshape(3 * hidden, hidden),
    }
    for index in range(num_experts):
        # Distinct values everywhere, so a shard that lands in the wrong row or
        # column cannot pass by coincidence.
        tensors[f"experts.{index}.gate_and_up_proj.weight"] = (
            torch.arange(2 * intermediate * hidden, dtype=torch.float32).reshape(
                2 * intermediate, hidden
            )
            + 1000.0 * index
        )
        tensors[f"experts.{index}.down_proj.weight"] = (
            torch.arange(hidden * intermediate, dtype=torch.float32).reshape(
                hidden, intermediate
            )
            + 1000.0 * index
            + 0.5
        )
    save_file(tensors, str(directory / "model.safetensors"))
    (directory / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {name: "model.safetensors" for name in tensors}}
        )
    )
    return tensors


def _expert_index(name: str) -> int | None:
    parts = name.split(".")
    if "experts" not in parts:
        return None
    return int(parts[parts.index("experts") + 1])


def test_expert_parallel_ranks_skip_experts_they_do_not_own(tmp_path: Path) -> None:
    hidden, intermediate, num_experts = 4, 6, 4
    reference = _write_checkpoint(
        tmp_path, hidden=hidden, intermediate=intermediate, num_experts=num_experts
    )
    owned = [2, 3]
    model = _Layer(hidden, intermediate, owned)

    def skip(name: str) -> bool:
        index = _expert_index(name)
        return index is not None and index not in owned

    missing, unexpected = load_tensor_parallel_checkpoint(
        model, tmp_path, skip_checkpoint_parameter=skip
    )

    assert missing == []
    assert unexpected == []
    torch.testing.assert_close(
        model.experts["2"].gate_and_up_proj.weight,
        reference["experts.2.gate_and_up_proj.weight"],
    )
    torch.testing.assert_close(
        model.experts["3"].down_proj.weight,
        reference["experts.3.down_proj.weight"],
    )


def test_unowned_experts_are_reported_without_a_skip_predicate(tmp_path: Path) -> None:
    hidden, intermediate = 4, 6
    _write_checkpoint(tmp_path, hidden=hidden, intermediate=intermediate, num_experts=4)
    model = _Layer(hidden, intermediate, [0])

    with pytest.raises(RuntimeError, match="unexpected"):
        load_tensor_parallel_checkpoint(model, tmp_path)


def test_two_rank_placement_reassembles_the_released_tensors(tmp_path: Path) -> None:
    hidden, intermediate, degree = 8, 6, 2
    reference = _write_checkpoint(
        tmp_path, hidden=hidden, intermediate=intermediate, num_experts=2
    )
    qkv_shards: list[torch.Tensor] = []
    gate_shards: list[torch.Tensor] = []
    up_shards: list[torch.Tensor] = []
    down_shards: list[torch.Tensor] = []
    for rank in range(degree):
        topology = TensorParallelTopology(
            ranks=tuple(range(degree)),
            rank=rank,
            rank_in_group=rank,
            degree=degree,
            # Loading never reduces, so a placeholder group is enough here.
            process_group=object(),
        )
        with use_tensor_parallel_topology(topology):
            model = _Layer(hidden, intermediate, [0])
            missing, unexpected = load_tensor_parallel_checkpoint(
                model,
                tmp_path,
                skip_checkpoint_parameter=lambda name: _expert_index(name)
                not in (None, 0),
            )
        assert (missing, unexpected) == ([], [])
        expert = model.experts["0"]
        gate, up = expert.gate_and_up_proj.weight.chunk(2)
        qkv_shards.append(model.qkv_proj.weight)
        gate_shards.append(gate)
        up_shards.append(up)
        down_shards.append(expert.down_proj.weight)

    expected_gate, expected_up = reference["experts.0.gate_and_up_proj.weight"].chunk(2)
    # The union of the rank-local shards has to be exactly the released tensor,
    # and the merged gate/up parameter has to stay splittable per rank.
    torch.testing.assert_close(
        torch.cat(qkv_shards, dim=0), reference["qkv_proj.weight"]
    )
    torch.testing.assert_close(torch.cat(gate_shards, dim=0), expected_gate)
    torch.testing.assert_close(torch.cat(up_shards, dim=0), expected_up)
    torch.testing.assert_close(
        torch.cat(down_shards, dim=1), reference["experts.0.down_proj.weight"]
    )


def test_merged_gate_up_shard_keeps_both_halves_on_every_rank(tmp_path: Path) -> None:
    hidden, intermediate = 4, 6
    reference = _write_checkpoint(
        tmp_path, hidden=hidden, intermediate=intermediate, num_experts=1
    )
    # A single-rank load must reproduce the checkpoint layout exactly, which is
    # the invariant the sharded loader narrows per logical half.
    model = _Layer(hidden, intermediate, [0])
    load_tensor_parallel_checkpoint(
        model,
        tmp_path,
        skip_checkpoint_parameter=lambda name: _expert_index(name) not in (None, 0),
    )
    merged = model.experts["0"].gate_and_up_proj
    assert merged.output_sizes == (intermediate, intermediate)
    assert merged.output_partition_sizes == (intermediate, intermediate)
    torch.testing.assert_close(
        merged.weight, reference["experts.0.gate_and_up_proj.weight"]
    )
