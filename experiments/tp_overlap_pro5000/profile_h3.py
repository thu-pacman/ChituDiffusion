from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from chitu_diffusion.models.minimax_h3.loader import load_minimax_h3_transformer
from chitu_diffusion.parallel.cp.context import EpeParallelContext
from chitu_diffusion.parallel.tp.linear import RowParallelLinear


@dataclass
class _RowProfile:
    name: str
    shape: tuple[int, ...]
    gemm_start: torch.cuda.Event
    gemm_end: torch.cuda.Event
    reduce_start: torch.cuda.Event
    reduce_end: torch.cuda.Event


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="/dockerdata/MiniMax-H3/FL2VA/transformer",
    )
    parser.add_argument("--tokens", type=int, default=21_824)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


def main() -> None:
    args = _args()
    context = EpeParallelContext.from_torchrun(
        allowed_widths=(1, 2),
        tensor_parallel_degree=4,
        ulysses_degree=2,
    )
    cp_world = context.cp_world_size
    cp_rank = context.cp_rank
    plane_start = context.tp_plane * cp_world
    lane = tuple(range(plane_start, plane_start + cp_world))
    with context.activate(lane):
        usp_topology = context.active_usp

    device = torch.device("cuda", context.local_rank)
    model = load_minimax_h3_transformer(
        args.model_path,
        device=device,
        attention_backend="fa4",
    )
    local_tokens = args.tokens // cp_world
    row_start = cp_rank * local_tokens
    hidden = torch.sin(
        (
            torch.arange(
                local_tokens * model.config.hidden_size,
                device=device,
                dtype=torch.float32,
            )
            + row_start * model.config.hidden_size
        ).reshape(local_tokens, model.config.hidden_size)
        * 0.001
    ).to(torch.bfloat16)
    position_ids = torch.zeros(args.tokens, 3, dtype=torch.float64, device=device)
    inverse_indices = torch.zeros(args.tokens, dtype=torch.long, device=device)
    token_tags = torch.zeros(args.tokens, dtype=torch.long, device=device)
    cu_seqlens = torch.tensor([0, args.tokens], dtype=torch.int32, device=device)

    def run_forward() -> tuple[torch.Tensor, torch.Tensor]:
        return model.forward_hidden(
            hidden,
            unique_timesteps=torch.tensor([1.0], device=device),
            inverse_indices=inverse_indices,
            token_tags=token_tags,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=args.tokens,
            ulysses_topology=usp_topology,
        )

    with torch.inference_mode():
        run_forward()
        torch.cuda.synchronize(device)

    names = {id(module): name for name, module in model.named_modules()}
    records: list[_RowProfile] = []
    original_forward = RowParallelLinear.forward

    def profiled_forward(self, inputs):
        if not self.input_is_parallel and self.tp_size > 1:
            inputs = inputs.chunk(self.tp_size, dim=-1)[self.tp_rank].contiguous()
        gemm_start, gemm_end = _event(), _event()
        reduce_start, reduce_end = _event(), _event()
        gemm_start.record()
        output = F.linear(inputs, self.weight, None)
        gemm_end.record()
        reduce_start.record()
        if self.reduce_results and self.tp_size > 1:
            dist.all_reduce(output, group=self.tp_group)
        reduce_end.record()
        if not self.skip_bias_add and self.bias is not None:
            output = output + self.bias
        records.append(
            _RowProfile(
                name=names.get(id(self), type(self).__name__),
                shape=tuple(output.shape),
                gemm_start=gemm_start,
                gemm_end=gemm_end,
                reduce_start=reduce_start,
                reduce_end=reduce_end,
            )
        )
        return output, self.bias if self.skip_bias_add else None

    RowParallelLinear.forward = profiled_forward
    try:
        with torch.inference_mode():
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            run_forward()
            torch.cuda.synchronize(device)
            wall_ms = (time.perf_counter() - started) * 1000.0
    finally:
        RowParallelLinear.forward = original_forward

    rows = []
    for record in records:
        gemm_ms = record.gemm_start.elapsed_time(record.gemm_end)
        reduce_ms = record.reduce_start.elapsed_time(record.reduce_end)
        rows.append(
            {
                "name": record.name,
                "shape": record.shape,
                "gemm_ms": gemm_ms,
                "all_reduce_ms": reduce_ms,
                "ideal_overlap_ms": min(gemm_ms, reduce_ms),
                "two_chunk_overlap_ms": 0.5 * min(gemm_ms, reduce_ms),
            }
        )
    gemm_ms = sum(row["gemm_ms"] for row in rows)
    all_reduce_ms = sum(row["all_reduce_ms"] for row in rows)
    ideal_overlap_ms = sum(row["ideal_overlap_ms"] for row in rows)
    two_chunk_overlap_ms = sum(row["two_chunk_overlap_ms"] for row in rows)
    local_report = {
        "rank": context.rank,
        "tp_rank": context.tensor_parallel.rank_in_group,
        "cp_rank": cp_rank,
        "wall_ms": wall_ms,
        "row_parallel_calls": len(rows),
        "gemm_ms": gemm_ms,
        "all_reduce_ms": all_reduce_ms,
        "all_reduce_fraction": all_reduce_ms / wall_ms,
        "two_chunk_estimated_wall_ms": wall_ms - two_chunk_overlap_ms,
        "two_chunk_estimated_speedup": wall_ms / (wall_ms - two_chunk_overlap_ms),
        "ideal_estimated_wall_ms": wall_ms - ideal_overlap_ms,
        "ideal_estimated_speedup": wall_ms / (wall_ms - ideal_overlap_ms),
        "rows": rows,
    }
    gathered: list[dict | None] = [None] * context.world_size
    dist.all_gather_object(gathered, local_report)
    if context.rank == 0:
        reports = [report for report in gathered if report is not None]
        critical = max(reports, key=lambda report: report["wall_ms"])
        result = {
            "tokens": args.tokens,
            "tp": 4,
            "cp": 2,
            "warmup_forwards": 1,
            "critical_rank": critical["rank"],
            "critical": critical,
            "per_rank": reports,
            "estimate_assumptions": (
                "GEMM and all-reduce scale linearly by chunk; no launch, "
                "contention, or synchronization overhead."
            ),
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        summary = {key: value for key, value in result.items() if key != "per_rank"}
        summary["critical"] = {
            key: value for key, value in critical.items() if key != "rows"
        }
        print(json.dumps(summary))
    context.close()


if __name__ == "__main__":
    main()
