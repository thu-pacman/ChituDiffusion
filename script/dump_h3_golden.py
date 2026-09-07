from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump SGLang MiniMax-H3 packed-layout and QKV contract golden data."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-length", type=int, default=32)
    parser.add_argument("--latent-t", type=int, default=2)
    parser.add_argument("--latent-h", type=int, default=8)
    parser.add_argument("--latent-w", type=int, default=8)
    parser.add_argument("--audio-t", type=int, default=3)
    args = parser.parse_args()

    from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
        _reorder_grouped_qkv_to_qkv,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
        minimax_h3_packed_sequence,
    )

    layout = minimax_h3_packed_sequence(
        text_len=args.text_length,
        latent_t=args.latent_t,
        latent_h=args.latent_h,
        latent_w=args.latent_w,
        audio_t=args.audio_t,
        include_keyframe_cond=False,
    )
    generator = torch.Generator().manual_seed(20260819)
    grouped_qkv = torch.randn(56 * 3 * 128, 7, generator=generator)
    reordered_qkv = _reorder_grouped_qkv_to_qkv(
        grouped_qkv,
        num_query_groups=56,
        heads_per_group=1,
        head_dim=128,
    )
    payload = {
        "arguments": vars(args),
        "layout": {
            key: value
            for key, value in layout.items()
            if torch.is_tensor(value) or isinstance(value, int)
        },
        "grouped_qkv": grouped_qkv,
        "reordered_qkv": reordered_qkv,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

