from __future__ import annotations

import argparse

import torch

from chitu_diffusion.models.minimax_h3 import (
    build_packed_sequence,
    reorder_grouped_qkv_to_qkv,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", required=True)
    parser.add_argument("--sglang-layers")
    parser.add_argument("--chitu-layers")
    parser.add_argument("--normalized-mae-threshold", type=float, default=0.05)
    args = parser.parse_args()
    payload = torch.load(args.golden, map_location="cpu", weights_only=True)
    arguments = payload["arguments"]
    actual = build_packed_sequence(
        text_length=arguments["text_length"],
        latent_t=arguments["latent_t"],
        latent_h=arguments["latent_h"],
        latent_w=arguments["latent_w"],
        audio_t=arguments["audio_t"],
    )
    golden = payload["layout"]
    mapping = {
        "seq_len": actual.sequence_length,
        "img_pos": actual.img_pos,
        "audio_pos": actual.audio_pos,
        "text_pos": actual.text_pos,
        "update_mask": actual.update_mask,
        "img_position_ids": actual.position_ids,
        "token_tags": actual.token_tags,
        "cu_seqlens": actual.cu_seqlens,
    }
    for name, value in mapping.items():
        expected = golden[name]
        if torch.is_tensor(value):
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
        elif value != expected:
            raise AssertionError(f"{name}: {value} != {expected}")
    reordered = reorder_grouped_qkv_to_qkv(
        payload["grouped_qkv"], num_heads=56, head_dim=128
    )
    torch.testing.assert_close(
        reordered, payload["reordered_qkv"], rtol=0, atol=0
    )
    print("SGLang parity passed: packed layout and grouped-QKV contract")
    if bool(args.sglang_layers) != bool(args.chitu_layers):
        raise ValueError("both --sglang-layers and --chitu-layers are required")
    if args.sglang_layers:
        sglang = torch.load(
            args.sglang_layers, map_location="cpu", weights_only=True
        )
        chitu = torch.load(
            args.chitu_layers, map_location="cpu", weights_only=True
        )
        if len(sglang["layers"]) != len(chitu["layers"]):
            raise AssertionError("layer count mismatch")
        ratios = []
        for expected, actual in zip(
            sglang["layers"], chitu["layers"], strict=True
        ):
            mae = (expected - actual).abs().float().mean()
            scale = expected.abs().float().mean().clamp_min(1e-6)
            ratios.append(float((mae / scale).item()))
        for name in ("video", "audio"):
            mae = (sglang[name] - chitu[name]).abs().float().mean()
            scale = sglang[name].abs().float().mean().clamp_min(1e-6)
            ratios.append(float((mae / scale).item()))
        worst = max(ratios)
        if worst > args.normalized_mae_threshold:
            raise AssertionError(
                f"normalized MAE {worst:.6f} exceeds "
                f"{args.normalized_mae_threshold:.6f}"
            )
        print(
            f"SGLang parity passed: 50 DiT layers and output heads "
            f"(worst normalized MAE={worst:.6f})"
        )


if __name__ == "__main__":
    main()

