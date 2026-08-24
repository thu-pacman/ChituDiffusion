from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist

from chitu_diffusion.models.minimax_h3 import load_minimax_h3_transformer
from chitu_diffusion.parallel import EpeParallelContext


def _token_grid(raw: str) -> tuple[int, ...]:
    values = tuple(int(value) for value in raw.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("token grid must not be empty")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--up", type=int, required=True)
    parser.add_argument("--backend", default="flex")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--token-grid", type=_token_grid)
    parser.add_argument("--used-tokens", type=int)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--output")
    parser.add_argument("--report")
    parser.add_argument("--save-layers", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    world_size = int(os.environ["WORLD_SIZE"])
    cp_world = world_size // args.tp
    widths = tuple(width for width in range(1, cp_world + 1) if cp_world % width == 0)
    context = EpeParallelContext.from_torchrun(
        allowed_widths=widths,
        tensor_parallel_degree=args.tp,
        ulysses_degree=cp_world,
    )
    cp_rank = context.rank % cp_world
    plane_start = context.tp_plane * cp_world
    lane_offset = (cp_rank // args.up) * args.up
    lane = tuple(
        range(plane_start + lane_offset, plane_start + lane_offset + args.up)
    )
    with context.activate(lane):
        active_usp = context.active_usp
    if active_usp.ulysses_degree != args.up or active_usp.ring_degree != 1:
        raise RuntimeError("requested lane is not a pure Ulysses topology")
    token_grid = args.token_grid or (args.tokens,)
    if any(
        tokens <= 0 or tokens % 64 or tokens % args.up
        for tokens in token_grid
    ):
        raise ValueError("token counts must be positive and divisible by 64 and UP")
    if args.used_tokens is not None and len(token_grid) != 1:
        raise ValueError("--used-tokens is only valid with one token count")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations positive")
    if args.save_layers and (args.warmup or args.iterations != 1):
        raise ValueError("--save-layers requires --warmup 0 --iterations 1")
    started = time.perf_counter()
    model = load_minimax_h3_transformer(
        args.model_path,
        device=f"cuda:{context.local_rank}",
        attention_backend=args.backend,
    )
    load_seconds = time.perf_counter() - started
    model_gib = torch.cuda.memory_allocated() / 1024**3
    torch.cuda.reset_peak_memory_stats()
    layer_outputs: list[torch.Tensor] = []
    hook_handles = []
    if args.save_layers:
        if args.tp != 1 or args.up != 1:
            raise ValueError("--save-layers currently requires TP1 x UP1")
        for block in model.blocks:
            hook_handles.append(
                block.register_forward_hook(
                    lambda module, inputs, output: layer_outputs.append(
                        output.detach().cpu()
                    )
                )
            )

    rows = []
    saved_video = None
    saved_audio = None
    for tokens in token_grid:
        used_tokens = (
            args.used_tokens
            if args.used_tokens is not None
            else tokens
        )
        if not 0 < used_tokens <= tokens:
            raise ValueError("--used-tokens must be in [1, tokens]")
        local_tokens = tokens // args.up
        row_start = cp_rank * local_tokens
        flat_start = row_start * model.config.hidden_size
        hidden = torch.sin(
            (
                torch.arange(
                    local_tokens * model.config.hidden_size,
                    device="cuda",
                    dtype=torch.float32,
                )
                + flat_start
            ).reshape(local_tokens, model.config.hidden_size)
            * 0.001
        ).to(torch.bfloat16)
        position_ids = torch.zeros(tokens, 3, dtype=torch.float64, device="cuda")
        inverse_indices = torch.zeros(tokens, dtype=torch.long, device="cuda")
        token_tags = torch.zeros(tokens, dtype=torch.long, device="cuda")
        cu_seqlens = torch.tensor(
            [0, used_tokens, tokens], dtype=torch.int32, device="cuda"
        )
        max_seqlen = max(used_tokens, tokens - used_tokens)

        def run_forward() -> tuple[torch.Tensor, torch.Tensor]:
            return model.forward_hidden(
                hidden,
                unique_timesteps=torch.tensor([1.0], device="cuda"),
                inverse_indices=inverse_indices,
                token_tags=token_tags,
                position_ids=position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                ulysses_topology=active_usp if args.up > 1 else None,
            )

        with torch.inference_mode():
            for _ in range(args.warmup):
                run_forward()
            torch.cuda.synchronize()
            forward_seconds = []
            for _ in range(args.iterations):
                torch.cuda.synchronize()
                forward_started = time.perf_counter()
                video, audio = run_forward()
                torch.cuda.synchronize()
                forward_seconds.append(time.perf_counter() - forward_started)
        video_full = model.gather_tp_output(video)
        audio_full = model.gather_tp_output(audio)
        checksum = torch.stack((video_full.float().sum(), audio_full.float().sum()))
        if args.save_layers and context.rank == 0:
            saved_video = video_full.cpu()
            saved_audio = audio_full.cpu()
        rows.append(
            {
                "tokens": tokens,
                "used_tokens": used_tokens,
                "forward_seconds": forward_seconds,
                "peak_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "checksum": checksum.cpu().tolist(),
            }
        )
        del hidden, position_ids, inverse_indices, token_tags, video, audio
        torch.cuda.empty_cache()
    if context.rank == 0:
        report = {
            "tp": args.tp,
            "up": args.up,
            "load_seconds": load_seconds,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "model_gib": model_gib,
            "rows": rows,
        }
        if args.report:
            with open(args.report, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
        if args.output:
            torch.save(
                {
                    "layers": layer_outputs,
                    "video": saved_video,
                    "audio": saved_audio,
                },
                args.output,
            )
        print(json.dumps(report))
    for handle in hook_handles:
        handle.remove()
    context.close()


if __name__ == "__main__":
    main()

