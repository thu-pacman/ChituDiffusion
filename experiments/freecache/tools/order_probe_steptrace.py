#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch


DEFAULT_QWEN_HQ_ANCHORS = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    10,
    12,
    14,
    16,
    18,
    20,
    22,
    25,
    28,
    31,
    34,
    38,
    42,
    46,
    48,
    49,
]


def load_payloads(root: Path) -> list[dict]:
    paths = sorted(root.rglob("flexcache_steptrace_vectors_*.pt"))
    if not paths:
        raise SystemExit(f"No flexcache_steptrace_vectors_*.pt files found under {root}")
    return [torch.load(path, map_location="cpu") for path in paths]


def select_trace(payloads: list[dict]) -> dict:
    by_cfg_rank = {int(payload.get("cfg_rank", int(payload.get("rank", 0)))): payload for payload in payloads}
    if 0 in by_cfg_rank and 1 in by_cfg_rank and "local_v" in by_cfg_rank[0] and "local_v" in by_cfg_rank[1]:
        cond = by_cfg_rank[0]
        uncond = by_cfg_rank[1]
        if list(cond["steps"]) != list(uncond["steps"]):
            raise SystemExit("CFG branch vector traces have mismatched steps.")
        return {
            "mode": "cfg_branch",
            "cond": cond,
            "uncond": uncond,
            "latents_pre": cond["latents_pre"],
            "latents": cond["latents"],
            "guided_v": cond["guided_v"],
            "sigmas_pre": cond["sigmas_pre"],
            "sigmas": cond["sigmas"],
            "steps": cond["steps"],
            "guidance_scale": float(cond.get("guidance_scale") or 1.0),
        }

    payload = payloads[0]
    return {
        "mode": "guided_only",
        "latents_pre": payload["latents_pre"],
        "latents": payload["latents"],
        "guided_v": payload["guided_v"],
        "sigmas_pre": payload["sigmas_pre"],
        "sigmas": payload["sigmas"],
        "steps": payload["steps"],
        "guidance_scale": float(payload.get("guidance_scale") or 1.0),
    }


def parse_steps(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def build_segments(anchors: list[int], total_steps: int, min_gap: int) -> list[tuple[int, int]]:
    anchors = sorted({step for step in anchors if 0 <= step < total_steps})
    return [(a, b) for a, b in zip(anchors, anchors[1:]) if b - a >= min_gap]


def rel_mse(current: torch.Tensor, reference: torch.Tensor) -> float:
    diff = (current - reference).to(torch.float64)
    ref = reference.to(torch.float64)
    return float(diff.square().mean().item() / max(ref.square().mean().item(), 1e-20))


def cosine(current: torch.Tensor, reference: torch.Tensor) -> float:
    current = current.to(torch.float64).flatten()
    reference = reference.to(torch.float64).flatten()
    return float(torch.dot(current, reference).item() / max(current.norm().item() * reference.norm().item(), 1e-20))


def jvp_predict(
    *,
    latents_pre: torch.Tensor,
    latents: torch.Tensor,
    velocities: torch.Tensor,
    sigmas_pre: torch.Tensor,
    sigmas: torch.Tensor,
    anchor_steps: list[int],
    anchor_position: int,
    target_step: int,
    order: int,
) -> torch.Tensor:
    anchor_index = anchor_steps[anchor_position]
    v_last = velocities[anchor_index].to(torch.float32)
    if order <= 0:
        return v_last

    actual_steps = min(max(1, order), anchor_position + 1)
    i0 = anchor_steps[anchor_position + 1 - actual_steps]
    i1 = anchor_index
    x_start = latents_pre[i0].to(torch.float32)
    x_end = latents[i1].to(torch.float32)
    s_start = sigmas_pre[i0].to(torch.float32)
    s_end = sigmas[i1].to(torch.float32)
    denom = s_end - s_start
    denom_b = denom.view(-1, *([1] * (x_start.ndim - denom.dim())))
    avg_u = (x_end - x_start) / denom_b
    inst_u = velocities[i0].to(torch.float32)
    jvp = (inst_u - avg_u) / denom_b
    delta_sigma = sigmas[target_step].to(torch.float32) - sigmas_pre[target_step].to(torch.float32)
    return v_last - jvp * delta_sigma.view(-1, *([1] * (v_last.ndim - delta_sigma.dim())))


def qwen_guidance_rescale(cond: torch.Tensor, uncond: torch.Tensor, guidance_scale: float) -> torch.Tensor:
    combined = uncond + float(guidance_scale) * (cond - uncond)
    cond_norm = torch.norm(cond, dim=-1, keepdim=True)
    noise_norm = torch.norm(combined, dim=-1, keepdim=True)
    return combined * (cond_norm / noise_norm.clamp_min(1e-20))


def predict_velocity(trace: dict, anchor_steps: list[int], anchor_position: int, target_step: int, order: int) -> torch.Tensor:
    if trace["mode"] == "cfg_branch":
        cond = jvp_predict(
            latents_pre=trace["cond"]["latents_pre"].to(torch.float32),
            latents=trace["cond"]["latents"].to(torch.float32),
            velocities=trace["cond"]["local_v"].to(torch.float32),
            sigmas_pre=trace["cond"]["sigmas_pre"].to(torch.float32),
            sigmas=trace["cond"]["sigmas"].to(torch.float32),
            anchor_steps=anchor_steps,
            anchor_position=anchor_position,
            target_step=target_step,
            order=order,
        )
        uncond = jvp_predict(
            latents_pre=trace["uncond"]["latents_pre"].to(torch.float32),
            latents=trace["uncond"]["latents"].to(torch.float32),
            velocities=trace["uncond"]["local_v"].to(torch.float32),
            sigmas_pre=trace["uncond"]["sigmas_pre"].to(torch.float32),
            sigmas=trace["uncond"]["sigmas"].to(torch.float32),
            anchor_steps=anchor_steps,
            anchor_position=anchor_position,
            target_step=target_step,
            order=order,
        )
        return qwen_guidance_rescale(cond, uncond, trace["guidance_scale"])

    return jvp_predict(
        latents_pre=trace["latents_pre"].to(torch.float32),
        latents=trace["latents"].to(torch.float32),
        velocities=trace["guided_v"].to(torch.float32),
        sigmas_pre=trace["sigmas_pre"].to(torch.float32),
        sigmas=trace["sigmas"].to(torch.float32),
        anchor_steps=anchor_steps,
        anchor_position=anchor_position,
        target_step=target_step,
        order=order,
    )


def replay_error(
    latents_pre: torch.Tensor,
    latents: torch.Tensor,
    velocities: torch.Tensor,
    sigmas_pre: torch.Tensor,
    sigmas: torch.Tensor,
) -> dict:
    worst = {"step": None, "rel_mse": 0.0}
    for step in range(len(velocities)):
        dt = (sigmas[step] - sigmas_pre[step]).to(torch.float32)
        pred = latents_pre[step].to(torch.float32) + dt.view(-1, *([1] * (latents_pre[step].ndim - dt.dim()))) * velocities[
            step
        ].to(torch.float32)
        error = rel_mse(pred, latents[step].to(torch.float32))
        if error > worst["rel_mse"]:
            worst = {"step": int(step), "rel_mse": error}
    return worst


def probe_segments(trace: dict, anchors: list[int], segments: list[tuple[int, int]], orders: list[int]) -> list[dict]:
    steps = [int(step) for step in trace["steps"]]
    if steps != list(range(len(steps))):
        raise SystemExit(f"Vector trace must contain contiguous zero-based steps, got {steps[:5]}...{steps[-5:]}")

    latents = trace["latents"].to(torch.float32)
    velocities = trace["guided_v"].to(torch.float32)
    sigmas_pre = trace["sigmas_pre"].to(torch.float32)
    sigmas = trace["sigmas"].to(torch.float32)

    rows = []
    anchor_positions = {step: idx for idx, step in enumerate(anchors)}
    for start, end in segments:
        if start not in anchor_positions:
            continue
        for order in orders:
            x_hat = latents[start].to(torch.float32)
            max_step_rel_mse = 0.0
            max_v_rel_mse = 0.0
            min_v_cosine = 1.0
            for step in range(start + 1, end + 1):
                v_hat = predict_velocity(trace, anchors, anchor_positions[start], step, order)
                max_v_rel_mse = max(max_v_rel_mse, rel_mse(v_hat, velocities[step]))
                min_v_cosine = min(min_v_cosine, cosine(v_hat, velocities[step]))
                dt = (sigmas[step] - sigmas_pre[step]).to(torch.float32)
                x_hat = x_hat + dt.view(-1, *([1] * (x_hat.ndim - dt.dim()))) * v_hat
                max_step_rel_mse = max(max_step_rel_mse, rel_mse(x_hat, latents[step]))

            rows.append(
                {
                    "start_step": start,
                    "end_step": end,
                    "gap": end - start,
                    "order": order,
                    "mode": trace["mode"],
                    "endpoint_latent_rel_mse": rel_mse(x_hat, latents[end]),
                    "max_latent_rel_mse": max_step_rel_mse,
                    "max_velocity_rel_mse": max_v_rel_mse,
                    "min_velocity_cosine": min_v_cosine,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict], output: Path, title: str) -> None:
    segments = sorted({(row["start_step"], row["end_step"]) for row in rows})
    labels = [f"{start}-{end}" for start, end in segments]
    orders = sorted({int(row["order"]) for row in rows})
    by_key = {(row["start_step"], row["end_step"], int(row["order"])): row for row in rows}

    fig, ax = plt.subplots(figsize=(12, 5))
    x = torch.arange(len(segments), dtype=torch.float32).numpy()
    width = 0.22
    offsets = {order: (idx - (len(orders) - 1) / 2) * width for idx, order in enumerate(orders)}
    for order in orders:
        values = [by_key[(start, end, order)]["endpoint_latent_rel_mse"] for start, end in segments]
        ax.bar(x + offsets[order], values, width=width, label=f"order {order}")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("endpoint latent relative MSE")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Closed-loop local order probe from StepTrace vector tensors.")
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--anchors", default="qwen_hq", help="comma-separated anchor steps or qwen_hq")
    parser.add_argument("--orders", default="0,1,2")
    parser.add_argument("--min-gap", type=int, default=2)
    parser.add_argument("--title", default="StepTrace local order probe")
    args = parser.parse_args()

    trace = select_trace(load_payloads(args.result_root))
    print(f"Using vector trace mode={trace['mode']} guidance_scale={trace['guidance_scale']}")
    total_steps = int(len(trace["steps"]))
    anchors = DEFAULT_QWEN_HQ_ANCHORS if args.anchors == "qwen_hq" else parse_steps(args.anchors)
    if not anchors:
        raise SystemExit("No anchors configured.")
    anchors = sorted({step for step in anchors if 0 <= step < total_steps})
    orders = parse_steps(args.orders) or [0, 1, 2]
    segments = build_segments(anchors, total_steps, args.min_gap)
    rows = probe_segments(trace, anchors, segments, orders)

    out_dir = args.result_root / "orderprobe"
    out_dir.mkdir(parents=True, exist_ok=True)
    replay = replay_error(
        trace["latents_pre"].to(torch.float32),
        trace["latents"].to(torch.float32),
        trace["guided_v"].to(torch.float32),
        trace["sigmas_pre"].to(torch.float32),
        trace["sigmas"].to(torch.float32),
    )
    (out_dir / "orderprobe_summary.json").write_text(json.dumps({"replay_error": replay, "rows": rows}, indent=2), encoding="utf-8")
    write_csv(out_dir / "orderprobe_summary.csv", rows)
    plot(rows, out_dir / "orderprobe_endpoint_latent_mse.png", args.title)
    print(out_dir / "orderprobe_endpoint_latent_mse.png")
    print(f"Replay worst step={replay['step']} rel_mse={replay['rel_mse']:.3e}")


if __name__ == "__main__":
    main()
