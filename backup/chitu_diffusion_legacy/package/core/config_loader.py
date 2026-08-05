from pathlib import Path
from typing import Iterable

from omegaconf import DictConfig, ListConfig, OmegaConf

from chitu_diffusion.core.utils import get_config_dir_path

DEFAULT_MODEL_NAME = "Wan2.1-T2V-1.3B"


def _config_dir() -> Path:
    return Path(get_config_dir_path())


def _load_yaml(path: Path) -> DictConfig:
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):
        raise ValueError(f"config file must contain a YAML mapping: {path}")
    return cfg


def _strip_legacy_defaults(cfg: DictConfig) -> DictConfig:
    if "defaults" not in cfg:
        return cfg
    cfg = cfg.copy()
    del cfg["defaults"]
    return cfg


def _default_model_name(base_cfg: DictConfig) -> str:
    defaults = base_cfg.get("defaults", [])
    if not isinstance(defaults, (list, ListConfig)):
        return ""

    for item in defaults:
        if not isinstance(item, (dict, DictConfig)):
            continue
        value = item.get("models")
        if value:
            return str(value)
    return DEFAULT_MODEL_NAME


def _split_model_override(overrides: Iterable[str]) -> tuple[str | None, list[str]]:
    model_name = None
    rest = []
    for override in overrides:
        if override.startswith("models="):
            model_name = override.split("=", 1)[1]
        else:
            rest.append(override)
    return model_name, rest


def _load_model_config(model_name: str) -> DictConfig:
    model_path = _config_dir() / "models" / f"{model_name}.yaml"
    return _load_yaml(model_path)


def load_config(overrides: Iterable[str] | None = None) -> DictConfig:
    """Load the runtime config.

    Supported CLI overrides use OmegaConf dotlist syntax, for
    example `models=Wan2.1-T2V-1.3B` and `models.ckpt_dir=/path/to/model`.
    """
    override_items = list(overrides or [])
    requested_model, remaining_overrides = _split_model_override(override_items)

    base_cfg = _load_yaml(_config_dir() / "serve_config.yaml")
    model_name = requested_model or _default_model_name(base_cfg)
    cfg = _strip_legacy_defaults(base_cfg)

    if model_name:
        cfg.models = _load_model_config(model_name)

    if remaining_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(remaining_overrides))

    validate_config(cfg)
    return cfg


def load_config_from_cli(argv: list[str] | None = None) -> DictConfig:
    import sys

    return load_config(sys.argv[1:] if argv is None else argv)


def _normalize_eval_types(raw_eval_type) -> list[str]:
    if raw_eval_type is None:
        return []
    if isinstance(raw_eval_type, str):
        value = raw_eval_type.strip().lower()
        return [] if value in {"", "none", "null"} else [value]
    if isinstance(raw_eval_type, (list, tuple, ListConfig)):
        values = []
        for item in raw_eval_type:
            value = str(item).strip().lower()
            if value in {"", "none", "null"}:
                continue
            values.append(value)
        return values
    raise ValueError(f"eval.eval_type must be string/list/null, got {type(raw_eval_type).__name__}")


def validate_config(config: DictConfig) -> None:
    port = int(config.serve.port)
    if not (1024 <= port <= 65535):
        raise ValueError(f"serve.port must be between 1024 and 65535, got {port}")

    num_blocks = int(config.infer.num_blocks)
    if num_blocks < 0 and num_blocks != -1:
        raise ValueError(f"infer.num_blocks must be positive or -1, got {num_blocks}")

    attn_type = str(config.infer.attn_type)
    allowed_attn = {
        "auto",
        "flash_attn",
        "flash",
        "flash2",
        "flash_v2",
        "fa2",
        "flashinfer",
        "flash_infer",
        "fi",
        "sage",
        "sparge",
        "sparse",
        "spas_sage",
        "torch_sdpa",
        "torch_sdpa_math",
        "sdpa",
        "sdpa_math",
        "torch",
        "torch_math",
        "ref",
    }
    if attn_type not in allowed_attn:
        raise ValueError(f"infer.attn_type must be one of {sorted(allowed_attn)}, got {attn_type}")

    tokenizer_type = str(config.models.tokenizer_type)
    if tokenizer_type not in {"hf", "tiktoken"}:
        raise ValueError(f"models.tokenizer_type must be one of ['hf', 'tiktoken'], got {tokenizer_type}")

    op_impl = str(config.infer.op_impl)
    if op_impl not in {"torch", "muxi_custom_kernel", "cpu"}:
        raise ValueError(f"infer.op_impl must be one of ['torch', 'muxi_custom_kernel', 'cpu'], got {op_impl}")

    bind_process_to_cpu = str(config.infer.bind_process_to_cpu)
    if bind_process_to_cpu not in {"auto", "none", "numa"}:
        raise ValueError(f"infer.bind_process_to_cpu must be one of ['auto', 'none', 'numa'], got {bind_process_to_cpu}")

    bind_thread_to_cpu = str(config.infer.bind_thread_to_cpu)
    if bind_thread_to_cpu not in {"physical_core", "logical_core"}:
        raise ValueError(
            f"infer.bind_thread_to_cpu must be one of ['physical_core', 'logical_core'], got {bind_thread_to_cpu}"
        )

    allowed_eval = {"fid", "fvd", "psnr", "ssim", "lpips"}
    invalid_eval = [item for item in _normalize_eval_types(config.eval.eval_type) if item not in allowed_eval]
    if invalid_eval:
        raise ValueError(f"eval.eval_type contains invalid items {invalid_eval}, allowed: {sorted(allowed_eval)}")

    output_root = str(config.output.root_dir).strip()
    if not output_root:
        raise ValueError("output.root_dir must be a non-empty path")

    if not isinstance(config.output.timer, bool):
        raise ValueError("output.timer must be bool")
    if not isinstance(config.output.run_log, bool):
        raise ValueError("output.run_log must be bool")
    if not isinstance(config.output.memory, bool):
        raise ValueError("output.memory must be bool")

    log_ranks = config.output.log_ranks
    if isinstance(log_ranks, str):
        if log_ranks.strip().lower() not in {"all", "*"}:
            for item in log_ranks.replace(";", ",").split(","):
                item = item.strip()
                if item:
                    int(item)
    elif isinstance(log_ranks, (list, tuple, ListConfig)):
        for item in log_ranks:
            if not isinstance(item, int) or item < 0:
                raise ValueError("output.log_ranks must contain non-negative integer ranks")
    else:
        raise ValueError("output.log_ranks must be a list of ranks or 'all'")
