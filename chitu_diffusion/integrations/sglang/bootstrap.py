from __future__ import annotations

from collections.abc import Sequence


H3_ARCHITECTURE = "MiniMaxH3DiTModel"
H3_ADAPTER = "chitu_diffusion.integrations.sglang.minimax_h3:MiniMaxH3DiTModel"


def install_sglang_h3_adapter() -> None:
    """Override SGLang's H3 DiT registry entry in the current process."""
    try:
        from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
    except ImportError as exc:
        raise RuntimeError(
            "SGLang diffusion is required to install the MiniMax-H3 adapter"
        ) from exc
    # This SGLang release's string-registration path constructs its internal
    # lazy record with an omitted component field.  Registering the class is
    # the public, process-local alternative and still leaves site-packages
    # untouched.
    from .minimax_h3 import MiniMaxH3DiTModel

    current = ModelRegistry.registered_models.get(H3_ARCHITECTURE)
    if getattr(current, "model_cls", None) is MiniMaxH3DiTModel:
        return
    ModelRegistry.register_model(H3_ARCHITECTURE, MiniMaxH3DiTModel)


def validate_launcher_args(argv: Sequence[str]) -> None:
    """Reject deployment modes for which the adapter has no exact contract."""
    args = list(argv)
    unsupported_flags = {
        "--use-fsdp-inference": "FSDP",
        "--enable-breakable-cuda-graph": "breakable CUDA graphs",
        "--quantization": "quantized DiT loading",
        "--transformer-weights-path": "alternate/quantized transformer weights",
    }
    for flag, feature in unsupported_flags.items():
        if flag in args:
            index = args.index(flag)
            next_value = args[index + 1].lower() if index + 1 < len(args) else ""
            if next_value not in {"false", "0", "no", "off"}:
                raise ValueError(
                    f"Chitu MiniMax-H3 adapter does not support {feature}: {flag}"
                )
    if "--model-variant" in args:
        index = args.index("--model-variant")
        if index + 1 >= len(args):
            raise ValueError("--model-variant requires a value")
        if args[index + 1].strip().lower() == "ref2va":
            raise ValueError(
                "Chitu MiniMax-H3 adapter does not support Ref2VA vision "
                "conditioning; use the SGLang-native DiT"
            )


__all__ = [
    "H3_ADAPTER",
    "H3_ARCHITECTURE",
    "install_sglang_h3_adapter",
    "validate_launcher_args",
]

