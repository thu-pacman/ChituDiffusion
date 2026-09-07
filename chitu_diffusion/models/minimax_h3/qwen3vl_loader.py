from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch import nn

from .qwen3vl_config import MiniMaxH3Qwen3VLConfig
from .qwen3vl_encoder import MiniMaxH3Qwen3VLEncoder

_PREFIX = "model.language_model"


def _checkpoint_map(root: Path) -> dict[str, Path]:
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        return {
            name: root / filename
            for name, filename in payload["weight_map"].items()
        }
    files = sorted(root.glob("*.safetensors"))
    if len(files) != 1:
        raise FileNotFoundError(
            f"expected one safetensors file or model index under {root}"
        )
    with safe_open(files[0], framework="pt", device="cpu") as handle:
        return {name: files[0] for name in handle.keys()}


def _replace_parameter(
    module: nn.Module, name: str, tensor: torch.Tensor
) -> nn.Parameter:
    previous = getattr(module, name)
    parameter = nn.Parameter(tensor, requires_grad=previous.requires_grad)
    setattr(module, name, parameter)
    return parameter


def _parameter_owner(root: nn.Module, name: str) -> tuple[nn.Module, str]:
    module_name, _, leaf = name.rpartition(".")
    module = dict(root.named_modules())[module_name]
    return module, leaf


class _ShardReader:
    def __init__(self, root: Path) -> None:
        self.weight_map = _checkpoint_map(root)
        self.stack = ExitStack()
        self.handles: dict[Path, Any] = {}

    def __enter__(self) -> _ShardReader:
        for path in sorted(set(self.weight_map.values())):
            self.handles[path] = self.stack.enter_context(
                safe_open(path, framework="pt", device="cpu")
            )
        return self

    def __exit__(self, *args: Any) -> None:
        self.stack.close()

    def slice(self, name: str) -> Any:
        try:
            path = self.weight_map[name]
        except KeyError as exc:
            raise KeyError(f"missing Qwen3-VL checkpoint tensor {name!r}") from exc
        return self.handles[path].get_slice(name)

    def full(
        self, name: str, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        return self.slice(name)[:].to(device=device, dtype=dtype)

    def rows(
        self,
        name: str,
        start: int,
        length: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self.slice(name)[start : start + length, :].to(
            device=device, dtype=dtype
        )

    def columns(
        self,
        name: str,
        start: int,
        length: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self.slice(name)[:, start : start + length].to(
            device=device, dtype=dtype
        )


@torch.no_grad()
def load_minimax_h3_qwen3vl_weights(
    model: MiniMaxH3Qwen3VLEncoder,
    checkpoint_dir: str | Path,
    *,
    device: torch.device | str = "cuda",
) -> None:
    """Stream rank-local Qwen3-VL shards directly from safetensors."""

    target_device = torch.device(device)
    root = Path(checkpoint_dir)
    with _ShardReader(root) as reader:
        embedding = reader.full(
            f"{_PREFIX}.embed_tokens.weight",
            device=target_device,
            dtype=model.embed_tokens.weight.dtype,
        )
        _replace_parameter(model.embed_tokens, "weight", embedding)

        for index, layer in enumerate(model.layers):
            source = f"{_PREFIX}.layers.{index}"
            attention = layer.self_attn
            qkv = attention.qkv_proj
            qkv_weight = torch.empty(
                qkv.weight.shape, dtype=qkv.weight.dtype, device=target_device
            )
            local_offset = 0
            for projection, local_size in zip(
                ("q_proj", "k_proj", "v_proj"),
                qkv.output_partition_sizes,
                strict=True,
            ):
                shard = reader.rows(
                    f"{source}.self_attn.{projection}.weight",
                    qkv.tp_rank * local_size,
                    local_size,
                    device=target_device,
                    dtype=qkv.weight.dtype,
                )
                qkv_weight[local_offset : local_offset + local_size].copy_(shard)
                local_offset += local_size
            _replace_parameter(qkv, "weight", qkv_weight)

            o_proj = attention.o_proj
            o_weight = reader.columns(
                f"{source}.self_attn.o_proj.weight",
                o_proj.tp_rank * o_proj.input_size_per_partition,
                o_proj.input_size_per_partition,
                device=target_device,
                dtype=o_proj.weight.dtype,
            )
            _replace_parameter(o_proj, "weight", o_weight)
            for norm_name in ("q_norm", "k_norm"):
                norm = getattr(attention, norm_name)
                weight = reader.full(
                    f"{source}.self_attn.{norm_name}.weight",
                    device=target_device,
                    dtype=norm.weight.dtype,
                )
                _replace_parameter(norm, "weight", weight)

            gate_up = layer.mlp.gate_up_proj
            gate_up_weight = torch.empty(
                gate_up.weight.shape,
                dtype=gate_up.weight.dtype,
                device=target_device,
            )
            local_offset = 0
            for projection, local_size in zip(
                ("gate_proj", "up_proj"),
                gate_up.output_partition_sizes,
                strict=True,
            ):
                shard = reader.rows(
                    f"{source}.mlp.{projection}.weight",
                    gate_up.tp_rank * local_size,
                    local_size,
                    device=target_device,
                    dtype=gate_up.weight.dtype,
                )
                gate_up_weight[
                    local_offset : local_offset + local_size
                ].copy_(shard)
                local_offset += local_size
            _replace_parameter(gate_up, "weight", gate_up_weight)

            down_proj = layer.mlp.down_proj
            down_weight = reader.columns(
                f"{source}.mlp.down_proj.weight",
                down_proj.tp_rank * down_proj.input_size_per_partition,
                down_proj.input_size_per_partition,
                device=target_device,
                dtype=down_proj.weight.dtype,
            )
            _replace_parameter(down_proj, "weight", down_weight)

            for norm_name in (
                "input_layernorm",
                "post_attention_layernorm",
            ):
                norm = getattr(layer, norm_name)
                weight = reader.full(
                    f"{source}.{norm_name}.weight",
                    device=target_device,
                    dtype=norm.weight.dtype,
                )
                _replace_parameter(norm, "weight", weight)

        if model.visual is not None:
            # Vision is intentionally replicated. Reading each tensor directly
            # from its safetensors slice avoids a second full state dict.
            for name, parameter in list(model.visual.named_parameters()):
                tensor = reader.full(
                    f"model.visual.{name}",
                    device=target_device,
                    dtype=parameter.dtype,
                )
                owner, leaf = _parameter_owner(model.visual, name)
                _replace_parameter(owner, leaf, tensor)

    model.to(target_device)
    model.eval().requires_grad_(False)


def load_minimax_h3_qwen3vl_encoder(
    checkpoint_dir: str | Path,
    *,
    device: torch.device | str = "cuda",
) -> MiniMaxH3Qwen3VLEncoder:
    checkpoint = Path(checkpoint_dir)
    config = MiniMaxH3Qwen3VLConfig.from_pretrained(checkpoint)
    model = MiniMaxH3Qwen3VLEncoder(config, device="meta")
    load_minimax_h3_qwen3vl_weights(model, checkpoint, device=device)
    return model


__all__ = [
    "load_minimax_h3_qwen3vl_encoder",
    "load_minimax_h3_qwen3vl_weights",
]
