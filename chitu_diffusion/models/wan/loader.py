from __future__ import annotations

import json
from pathlib import Path

import torch
from accelerate import init_empty_weights
from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler
from diffusers.loaders.single_file_utils import (
    convert_wan_transformer_to_diffusers,
    convert_wan_vae_to_diffusers,
)
from safetensors.torch import load_file
from transformers import AutoTokenizer, UMT5Config, UMT5EncoderModel

from .transformer import EpeWanTransformer3DModel


def _umt5_xxl_config() -> UMT5Config:
    return UMT5Config(
        vocab_size=256384,
        d_model=4096,
        d_kv=64,
        d_ff=10240,
        num_layers=24,
        num_decoder_layers=24,
        num_heads=64,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        feed_forward_proj="gated-gelu",
        dense_act_fn="gelu_new",
        is_encoder_decoder=True,
        scalable_attention=True,
        tie_word_embeddings=False,
    )


def convert_wan_umt5_encoder_state_dict(
    checkpoint: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map the official Wan UMT5 checkpoint onto Transformers UMT5."""

    converted: dict[str, torch.Tensor] = {}
    token_embedding = checkpoint.pop("token_embedding.weight")
    converted["shared.weight"] = token_embedding
    converted["encoder.embed_tokens.weight"] = token_embedding
    for index in range(24):
        source = f"blocks.{index}"
        target = f"encoder.block.{index}"
        converted[f"{target}.layer.0.layer_norm.weight"] = checkpoint.pop(
            f"{source}.norm1.weight"
        )
        for projection in ("q", "k", "v", "o"):
            converted[f"{target}.layer.0.SelfAttention.{projection}.weight"] = (
                checkpoint.pop(f"{source}.attn.{projection}.weight")
            )
        converted[f"{target}.layer.0.SelfAttention.relative_attention_bias.weight"] = (
            checkpoint.pop(f"{source}.pos_embedding.embedding.weight")
        )
        converted[f"{target}.layer.1.layer_norm.weight"] = checkpoint.pop(
            f"{source}.norm2.weight"
        )
        converted[f"{target}.layer.1.DenseReluDense.wi_0.weight"] = checkpoint.pop(
            f"{source}.ffn.gate.0.weight"
        )
        converted[f"{target}.layer.1.DenseReluDense.wi_1.weight"] = checkpoint.pop(
            f"{source}.ffn.fc1.weight"
        )
        converted[f"{target}.layer.1.DenseReluDense.wo.weight"] = checkpoint.pop(
            f"{source}.ffn.fc2.weight"
        )
    converted["encoder.final_layer_norm.weight"] = checkpoint.pop("norm.weight")
    if checkpoint:
        raise ValueError(
            "unexpected Wan UMT5 checkpoint keys: " + ", ".join(sorted(checkpoint)[:8])
        )
    return converted


def load_wan_diffusers_components(
    model_path: str | Path,
    *,
    parallel_context,
    attention_mode: str,
    torch_dtype: torch.dtype,
    flow_shift: float,
    local_files_only: bool,
):
    """Load original Wan2.1 files into official Diffusers/Transformers modules."""

    root = Path(model_path)
    required = {
        "transformer": root / "diffusion_pytorch_model.safetensors",
        "vae": root / "Wan2.1_VAE.pth",
        "text_encoder": root / "models_t5_umt5-xxl-enc-bf16.pth",
        "tokenizer": root / "google" / "umt5-xxl",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Wan original checkpoint is missing components: {', '.join(missing)}"
        )

    config_path = root / "config.json"
    if not config_path.exists():
        raise FileNotFoundError("Wan original checkpoint is missing config.json")
    original_config = json.loads(config_path.read_text(encoding="utf-8"))
    if original_config.get("model_type") != "t2v":
        raise NotImplementedError("Wan EPE original loader currently supports T2V")
    model_dim = int(original_config["dim"])
    num_heads = int(original_config["num_heads"])
    if model_dim % num_heads:
        raise ValueError("Wan transformer dim must be divisible by num_heads")

    with init_empty_weights():
        transformer = EpeWanTransformer3DModel(
            patch_size=(1, 2, 2),
            num_attention_heads=num_heads,
            attention_head_dim=model_dim // num_heads,
            in_channels=int(original_config["in_dim"]),
            out_channels=int(original_config["out_dim"]),
            text_dim=4096,
            freq_dim=int(original_config["freq_dim"]),
            ffn_dim=int(original_config["ffn_dim"]),
            num_layers=int(original_config["num_layers"]),
            cross_attn_norm=True,
            qk_norm="rms_norm_across_heads",
            eps=float(original_config["eps"]),
        )
    transformer_state = convert_wan_transformer_to_diffusers(
        load_file(str(required["transformer"]), device="cpu")
    )
    transformer.load_state_dict(transformer_state, strict=True, assign=True)
    del transformer_state
    transformer.to(dtype=torch_dtype).eval().requires_grad_(False)
    transformer.configure_epe(parallel_context, attention_mode=attention_mode)

    with init_empty_weights():
        vae = AutoencoderKLWan()
    vae_checkpoint = torch.load(
        required["vae"], map_location="cpu", mmap=True, weights_only=True
    )
    vae_state = convert_wan_vae_to_diffusers(vae_checkpoint)
    vae.load_state_dict(vae_state, strict=True, assign=True)
    del vae_checkpoint, vae_state
    vae.to(dtype=torch.float32).eval().requires_grad_(False)

    with init_empty_weights():
        text_encoder = UMT5EncoderModel(_umt5_xxl_config())
    text_checkpoint = torch.load(
        required["text_encoder"],
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    text_state = convert_wan_umt5_encoder_state_dict(text_checkpoint)
    text_encoder.load_state_dict(text_state, strict=True, assign=True)
    del text_checkpoint, text_state
    text_encoder.tie_weights()
    text_encoder.to(dtype=torch.bfloat16).eval().requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(
        required["tokenizer"], local_files_only=local_files_only
    )
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=float(flow_shift),
        use_dynamic_shifting=False,
    )
    return tokenizer, text_encoder, vae, scheduler, transformer
