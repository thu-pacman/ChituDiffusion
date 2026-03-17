# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from hydra.experimental.callback import Callback
from omegaconf import DictConfig, ListConfig
import sys
from logging import getLogger

logger = getLogger(__name__)


class ServeConfigRules(Callback):
    def __init__(self) -> None:
        super().__init__()

    def _exit_with_error(self, message):
        """Fatal error, exit method"""
        logger.error(f"Config Error: {message}")
        sys.exit(1)

    def on_job_start(self, config: DictConfig, **kwargs) -> None:
        port = config.serve.port
        if not (1024 <= port <= 65535):
            self._exit_with_error(f"Port must be between 1024 and 65535, got {port}")

        num_blocks = config.infer.num_blocks
        if num_blocks < 0 and num_blocks != -1:
            self._exit_with_error(
                f"num_blocks must be positive or -1 (got {num_blocks})"
            )

        attn_type = config.infer.attn_type
        if attn_type == "npu":
            try:
                import torch_npu
            except ImportError:
                self._exit_with_error(
                    f"torch-npu required for attn_type=npu (got {attn_type})"
                )
        if attn_type not in {
            "auto",
            "flash_attn",
            "flash_mla",
            "flash_infer",
            "triton",
            "npu",
            "ref",
            "sage",
            "sparge"
            }:
            self._exit_with_error(
                f"attn_type must be one of [auto, flash_attn, flash_mla, flash_infer, triton, npu, ref, sage, sparge], got {attn_type}"
            )

        model_name = config.models.name
        model_type = config.models.type
        if attn_type == "flash_infer":
            if config.models.n_heads // config.models.n_kv_heads not in [1, 2, 3, 4, 8]:
                self._exit_with_error(
                    f"model {model_name} is not compatible with flash_infer"
                )
        elif attn_type == "flash_mla":
            if "deepseek-v3" not in model_type:
                self._exit_with_error(
                    f"model {model_name} is not compatible with flash_mla"
                )

        if model_type == "hf-gpt-oss" and attn_type != "ref":
            self._exit_with_error(f"model {model_name} is only compatible with ref")

        tokenizer_type = config.models.tokenizer_type
        if tokenizer_type not in {"hf", "tiktoken"}:
            self._exit_with_error(
                f"tokenizer_type must be one of [hf, tiktoken], got {tokenizer_type}"
            )

        op_impl = config.infer.op_impl
        if op_impl not in {"torch", "muxi_custom_kernel", "cpu"}:
            self._exit_with_error(
                f"op_impl must be one of [torch, muxi_custom_kernel, cpu], got {op_impl}"
            )

        bind_process_to_cpu = config.infer.bind_process_to_cpu
        if bind_process_to_cpu not in {"auto", "none", "numa"}:
            self._exit_with_error(
                f"bind_process_to_cpu must be one of [auto, none, numa], got {bind_process_to_cpu}"
            )

        bind_thread_to_cpu = config.infer.bind_thread_to_cpu
        if bind_thread_to_cpu not in {"physical_core", "logical_core"}:
            self._exit_with_error(
                f"bind_thread_to_cpu must be one of [physical_core, logical_core], got {bind_thread_to_cpu}"
            )

        allowed_eval_types = {"vbench", "fid", "fvd", "psnr", "ssim", "lpips"}
        raw_eval_type = config.eval.eval_type
        if raw_eval_type is None:
            normalized_eval_types = []
        elif isinstance(raw_eval_type, str):
            value = raw_eval_type.strip().lower()
            if value in {"", "none", "null"}:
                normalized_eval_types = []
            else:
                normalized_eval_types = [value]
        elif isinstance(raw_eval_type, (list, tuple, ListConfig)):
            normalized_eval_types = []
            for item in raw_eval_type:
                value = str(item).strip().lower()
                if value in {"", "none", "null"}:
                    continue
                normalized_eval_types.append(value)
        else:
            self._exit_with_error(
                f"eval.eval_type must be string/list/null, got {type(raw_eval_type).__name__}"
            )

        invalid_eval_types = [
            item for item in normalized_eval_types if item not in allowed_eval_types
        ]
        if invalid_eval_types:
            self._exit_with_error(
                f"eval.eval_type contains invalid items {invalid_eval_types}, allowed: {sorted(allowed_eval_types)}"
            )

        output_root_dir = str(config.output.root_dir).strip()
        if not output_root_dir:
            self._exit_with_error("output.root_dir must be a non-empty path")

        if not isinstance(config.output.enable_timer_dump, bool):
            self._exit_with_error("output.enable_timer_dump must be bool")

        if not isinstance(config.output.enable_run_log, bool):
            self._exit_with_error("output.enable_run_log must be bool")
