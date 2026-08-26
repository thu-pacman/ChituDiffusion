from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29671")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
        MiniMaxH3DiTConfig,
    )
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
    )
    from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPAImpl
    from sglang.multimodal_gen.runtime.loader.fsdp_load import maybe_load_fsdp_model
    import sglang.multimodal_gen.runtime.models.dits.minimax_h3 as h3_impl
    from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
        MiniMaxH3Attention,
        MiniMaxH3DiTModel,
        _rope_cos_sin_cache,
    )

    # Disable SGLang's in-place rounding kernels so this dump is a portable
    # mathematical golden rather than a kernel-specific SM90 rounding trace.
    h3_impl._modulate_scale_shift = (
        lambda x, shift, scale, indices, *, dtype: (
            x * (1 + scale.index_select(0, indices))
            + shift.index_select(0, indices)
        ).to(dtype)
    )
    h3_impl._modulate_gate = (
        lambda x, gate, other, indices, *, dtype: (
            x + gate.index_select(0, indices) * other
        ).to(dtype)
    )
    h3_impl._silu_mul = lambda hidden, *, reuse_input: (
        F.silu(hidden.chunk(2, dim=-1)[0]) * hidden.chunk(2, dim=-1)[1]
    )
    h3_impl._apply_qk_norm = (
        lambda query, key, q_norm, k_norm, head_dim: (
            q_norm(query),
            k_norm(key),
        )
    )

    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
    model_path = Path(args.model_path)
    hf_config = json.loads((model_path / "config.json").read_text())
    hf_config.pop("_class_name", None)
    hf_config.pop("_diffusers_version", None)
    model = maybe_load_fsdp_model(
        model_cls=MiniMaxH3DiTModel,
        init_params={
            "config": MiniMaxH3DiTConfig(),
            "hf_config": hf_config,
            "quant_config": None,
        },
        weight_dir_list=[
            str(path) for path in sorted(model_path.glob("*.safetensors"))
        ],
        device=torch.device("cuda:0"),
        hsdp_replicate_dim=1,
        hsdp_shard_dim=1,
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        strict=False,
    ).eval()
    for module in model.modules():
        if isinstance(module, MiniMaxH3Attention):
            module._use_fused_qknorm_rope = False
            module._attention_impl = SDPAImpl(
                num_heads=module.num_heads,
                head_size=module.head_dim,
                causal=False,
                softmax_scale=module.softmax_scale,
            )

    hidden = torch.sin(
        torch.arange(
            64 * model.hidden_size, device="cuda", dtype=torch.float32
        ).reshape(64, model.hidden_size)
        * 0.001
    ).to(torch.bfloat16)
    timestep = torch.tensor([1.0], device="cuda")
    adaln_input = F.silu(model.time_embedder(timestep)).to(torch.bfloat16)
    combined = torch.zeros(64, dtype=torch.long, device="cuda")
    inverse = torch.zeros_like(combined)
    cu = torch.tensor([0, 63, 64], dtype=torch.int32, device="cuda")
    rope_freqs = model.rope(
        torch.zeros(1, 64, 3, dtype=torch.float64, device="cuda")
    )
    rope_cache = (
        _rope_cos_sin_cache(rope_freqs, dtype=torch.bfloat16),
        torch.arange(64, device="cuda", dtype=torch.long),
    )
    layers = []
    with torch.inference_mode():
        for block in model.blocks:
            hidden = block(
                hidden,
                adaln_input=adaln_input,
                combined_indices=combined,
                rope_cache=rope_cache,
                cu_seqlens=cu,
                cu_seqlens_host=(0, 63, 64),
                max_seqlen=63,
            )
            layers.append(hidden.cpu())
        video, audio = model.final_layer(
            hidden, adaln_input=adaln_input, inverse_indices=inverse
        )
    torch.save(
        {
            "layers": layers,
            "video": video.cpu(),
            "audio": audio.cpu(),
        },
        args.output,
    )
    print(f"wrote {args.output}")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

