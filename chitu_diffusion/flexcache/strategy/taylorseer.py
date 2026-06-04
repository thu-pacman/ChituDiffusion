import functools
import math
import os
from logging import getLogger
from typing import Any, Dict, Iterable, Optional

import torch
import torch.amp as amp

from chitu_diffusion.core.distributed.parallel_state import get_cp_group
from chitu_diffusion.core.models.backbone import (
    add_backbone_values,
    detach_backbone_value,
    scale_backbone_value,
    sub_backbone_values,
)
from chitu_diffusion.flexcache.flexcache_manager import FlexCacheStrategy
from chitu_diffusion.runtime.backend import CFGType, DiffusionBackend
from chitu_diffusion.runtime.output_layout import debug_output_dir

logger = getLogger(__name__)


def _taylor_formula(derivative_dict: Dict[int, torch.Tensor], distance: int) -> torch.Tensor:
    output = None
    for order in sorted(derivative_dict):
        term = scale_backbone_value(derivative_dict[order], (distance ** order) / math.factorial(order))
        output = term if output is None else add_backbone_values(output, term)
    if output is None:
        raise ValueError("TaylorSeer cache is empty for the requested module.")
    return output


class TaylorSeerStrategy(FlexCacheStrategy):
    """TaylorSeer module-output cache adapted to the FlexCache strategy API."""

    def __init__(
        self,
        task,
        fresh_threshold: int = 5,
        max_order: int = 1,
        first_enhance: int = 1,
        warmup_steps: int = 0,
        cooldown_steps: int = 0,
    ):
        super().__init__()
        self.type = "taylorseer"
        self.tradeoff_score = 1.0 / max(1, int(fresh_threshold))
        self.num_steps = int(task.req.params.num_inference_steps)
        self.fresh_threshold = max(1, int(fresh_threshold))
        self.max_order = max(0, int(max_order))
        self.first_enhance = max(0, int(first_enhance))
        self.warmup_steps = max(0, int(warmup_steps))
        self.cooldown_steps = max(0, int(cooldown_steps))
        self._branch_states: Dict[str, Dict[str, Any]] = {}
        self._wrapped_blocks = []
        self._vis_records: Dict[int, int] = {}
        self._vis_max_step = -1

    def get_reuse_key(self, **kwargs) -> Optional[Any]:
        if self._current_type_for_step() != "Taylor":
            return None
        return self._cache_key(**kwargs)

    def reuse(self, cached_feature: Dict[int, torch.Tensor], distance: int, **kwargs) -> torch.Tensor:
        return _taylor_formula(cached_feature, distance)

    def get_store_key(self, **kwargs) -> Optional[Any]:
        return self._cache_key(**kwargs)

    def store(self, fresh_feature: torch.Tensor, **kwargs) -> Dict[int, torch.Tensor]:
        key = self._cache_key(**kwargs)
        previous = DiffusionBackend.flexcache.cache.get(key, {})
        updated: Dict[int, torch.Tensor] = {0: detach_backbone_value(fresh_feature)}
        activated_steps = self._branch_state()["activated_steps"]
        distance = 1 if len(activated_steps) < 2 else max(1, activated_steps[-1] - activated_steps[-2])
        for order in range(self.max_order):
            if order not in previous or self._current_step_value() <= self.first_enhance - 2:
                break
            updated[order + 1] = scale_backbone_value(sub_backbone_values(updated[order], previous[order]), 1.0 / distance)
        return updated

    def wrap_module_with_strategy(self, module: torch.nn.Module) -> None:
        if not hasattr(module, "backbone_blocks"):
            raise ValueError(f"{module.__class__.__name__} does not implement the backbone block API.")

        wrapped = 0
        for block_info in module.backbone_blocks():
            block = block_info.module
            if self._wrap_flux_double_block(block_info.index, block):
                wrapped += 1
            elif self._wrap_flux1_single_block(block_info.index, block):
                wrapped += 1
            elif self._wrap_flux2_single_block(block_info.index, block):
                wrapped += 1
            elif self._wrap_wan_attention_block(block_info.index, block):
                wrapped += 1

        if wrapped == 0:
            raise ValueError(
                f"{module.__class__.__name__} does not expose supported TaylorSeer Flux/Wan blocks."
            )

        logger.info(
            "Module %s wrapped with TaylorSeer module strategy: blocks=%d fresh_threshold=%d max_order=%d warmup=%d cooldown=%d",
            module.__class__.__name__,
            wrapped,
            self.fresh_threshold,
            self.max_order,
            self.warmup_steps,
            self.cooldown_steps,
        )

    def unwrap_module(self, module: torch.nn.Module) -> None:
        for block in self._wrapped_blocks:
            if hasattr(block, "_taylorseer_original_forward"):
                block.forward = block._taylorseer_original_forward
                delattr(block, "_taylorseer_original_forward")

        run_output_dir = os.environ.get("CHITU_CURRENT_OUTPUT_DIR", "").strip()
        if run_output_dir:
            self._save_policy_ppm(debug_output_dir(run_output_dir))
            DiffusionBackend.flexcache.record_cache_memory(
                "flexcache_unwrap",
                task_id=getattr(DiffusionBackend.generator.current_task, "task_id", None),
            )
        logger.info("Module %s unwrapped from TaylorSeer strategy", module.__class__.__name__)

    def reset_state(self):
        self._branch_states = {}
        self._wrapped_blocks = []
        self._vis_records = {}
        self._vis_max_step = -1
        DiffusionBackend.flexcache.clear_cache()

    def _store_module(self, layer_idx: int, module_name: str, feature: torch.Tensor) -> None:
        key = self.get_store_key(layer=layer_idx, module=module_name)
        DiffusionBackend.flexcache.cache[key] = self.store(
            fresh_feature=feature,
            layer=layer_idx,
            module=module_name,
        )
        if layer_idx == 0:
            DiffusionBackend.flexcache.record_cache_memory(
                "flexcache_store",
                task_id=getattr(DiffusionBackend.generator.current_task, "task_id", None),
                extra={"cache_key": "taylorseer"},
            )

    def _reuse_module(self, layer_idx: int, module_name: str) -> torch.Tensor:
        return self.reuse(
            DiffusionBackend.flexcache.cache[self._cache_key(layer=layer_idx, module=module_name)],
            distance=self._distance_from_anchor(),
        )

    def _should_reuse_modules(self, layer_idx: int, module_names: Iterable[str]) -> bool:
        self._ensure_step_decision()
        return self._branch_state()["current_type"] == "Taylor" and all(
            self._has_module_cache(layer_idx, module_name) for module_name in module_names
        )

    def _store_and_record(self, layer_idx: int, module_name: str, feature: torch.Tensor) -> None:
        self._store_module(layer_idx, module_name, feature)
        self._record_compute("compute", layer_idx, module_name)
        self._record_step_policy(self._current_step_value(), 1)

    def _reuse_and_record(self, layer_idx: int, module_name: str) -> torch.Tensor:
        feature = self._reuse_module(layer_idx, module_name)
        self._record_compute("reuse", layer_idx, module_name)
        self._record_step_policy(self._current_step_value(), 2)
        return feature

    def _wrap_forward(self, block: torch.nn.Module, forward_fn) -> bool:
        if hasattr(block, "_taylorseer_original_forward"):
            return False
        block._taylorseer_original_forward = block.forward
        block.forward = forward_fn
        self._wrapped_blocks.append(block)
        return True

    def _wrap_flux1_single_block(self, layer_idx: int, block: torch.nn.Module) -> bool:
        required = ("norm", "proj_mlp", "act_mlp", "attn", "proj_out")
        if not all(hasattr(block, name) for name in required):
            return False
        original_forward = block.forward

        @functools.wraps(original_forward)
        def forward_with_taylorseer(hidden_states, temb, image_rotary_emb=None, joint_attention_kwargs=None):
            residual = hidden_states
            norm_hidden_states, gate = block.norm(hidden_states, emb=temb)
            joint_attention_kwargs = joint_attention_kwargs or {}
            module_name = "single_total"

            if self._should_reuse_modules(layer_idx, (module_name,)):
                hidden_states = self._reuse_and_record(layer_idx, module_name)
            else:
                mlp_hidden_states = block.act_mlp(block.proj_mlp(norm_hidden_states))
                attn_output = block.attn(
                    hidden_states=norm_hidden_states,
                    image_rotary_emb=image_rotary_emb,
                    **joint_attention_kwargs,
                )
                hidden_states = block.proj_out(torch.cat([attn_output, mlp_hidden_states], dim=2))
                self._store_and_record(layer_idx, module_name, hidden_states)

            hidden_states = residual + gate.unsqueeze(1) * hidden_states
            if hidden_states.dtype == torch.float16:
                hidden_states = hidden_states.clip(-65504, 65504)
            return hidden_states

        return self._wrap_forward(block, forward_with_taylorseer)

    def _wrap_flux2_single_block(self, layer_idx: int, block: torch.nn.Module) -> bool:
        required = ("norm", "attn")
        if not all(hasattr(block, name) for name in required):
            return False
        original_forward = block.forward

        @functools.wraps(original_forward)
        def forward_with_taylorseer(
            hidden_states,
            encoder_hidden_states,
            temb_mod,
            image_rotary_emb=None,
            joint_attention_kwargs=None,
            split_hidden_states=False,
            text_seq_len=None,
        ):
            if encoder_hidden_states is not None:
                text_seq_len = encoder_hidden_states.shape[1]
                hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

            mod_shift, mod_scale, mod_gate = self._flux2_split_modulation(temb_mod, 1)[0]
            norm_hidden_states = block.norm(hidden_states)
            norm_hidden_states = (1 + mod_scale) * norm_hidden_states + mod_shift
            joint_attention_kwargs = joint_attention_kwargs or {}
            module_name = "single_total"

            if self._should_reuse_modules(layer_idx, (module_name,)):
                attn_output = self._reuse_and_record(layer_idx, module_name)
            else:
                attn_output = block.attn(
                    hidden_states=norm_hidden_states,
                    image_rotary_emb=image_rotary_emb,
                    **joint_attention_kwargs,
                )
                self._store_and_record(layer_idx, module_name, attn_output)

            hidden_states = hidden_states + mod_gate * attn_output
            if hidden_states.dtype == torch.float16:
                hidden_states = hidden_states.clip(-65504, 65504)

            if split_hidden_states:
                encoder_hidden_states, hidden_states = hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]
                return encoder_hidden_states, hidden_states
            return hidden_states

        return self._wrap_forward(block, forward_with_taylorseer)

    def _wrap_flux_double_block(self, layer_idx: int, block: torch.nn.Module) -> bool:
        required = ("norm1", "norm1_context", "attn", "norm2", "ff", "norm2_context", "ff_context")
        if not all(hasattr(block, name) for name in required):
            return False
        original_forward = block.forward

        @functools.wraps(original_forward)
        def forward_with_taylorseer(
            hidden_states,
            encoder_hidden_states,
            *args,
            temb=None,
            temb_mod_img=None,
            temb_mod_txt=None,
            image_rotary_emb=None,
            joint_attention_kwargs=None,
        ):
            if args:
                if len(args) == 1:
                    temb = args[0]
                elif len(args) == 2:
                    temb_mod_img, temb_mod_txt = args
                else:
                    raise TypeError(f"Unexpected TaylorSeer Flux block positional args: {len(args)}")
            flux2 = temb is None and temb_mod_img is not None and temb_mod_txt is not None
            if flux2:
                (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = self._flux2_split_modulation(
                    temb_mod_img, 2
                )
                (c_shift_msa, c_scale_msa, c_gate_msa), (c_shift_mlp, c_scale_mlp, c_gate_mlp) = (
                    self._flux2_split_modulation(temb_mod_txt, 2)
                )
                norm_hidden_states = block.norm1(hidden_states)
                norm_hidden_states = (1 + scale_msa) * norm_hidden_states + shift_msa
                norm_encoder_hidden_states = block.norm1_context(encoder_hidden_states)
                norm_encoder_hidden_states = (1 + c_scale_msa) * norm_encoder_hidden_states + c_shift_msa
                gate_dims = None
            else:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.norm1(hidden_states, emb=temb)
                norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = block.norm1_context(
                    encoder_hidden_states, emb=temb
                )
                gate_dims = 1
            joint_attention_kwargs = joint_attention_kwargs or {}
            module_names = ("img_attn", "img_mlp", "txt_attn", "txt_mlp")

            if self._should_reuse_modules(layer_idx, module_names):
                attn_output = self._reuse_and_record(layer_idx, "img_attn")
                hidden_states = hidden_states + self._apply_gate(gate_msa, attn_output, gate_dims)
                ff_output = self._reuse_and_record(layer_idx, "img_mlp")
                hidden_states = hidden_states + self._apply_gate(gate_mlp, ff_output, gate_dims)

                context_attn_output = self._reuse_and_record(layer_idx, "txt_attn")
                encoder_hidden_states = encoder_hidden_states + self._apply_gate(
                    c_gate_msa, context_attn_output, gate_dims
                )
                context_ff_output = self._reuse_and_record(layer_idx, "txt_mlp")
                encoder_hidden_states = encoder_hidden_states + self._apply_gate(
                    c_gate_mlp, context_ff_output, gate_dims
                )
            else:
                attention_outputs = block.attn(
                    hidden_states=norm_hidden_states,
                    encoder_hidden_states=norm_encoder_hidden_states,
                    image_rotary_emb=image_rotary_emb,
                    **joint_attention_kwargs,
                )
                if len(attention_outputs) == 2:
                    attn_output, context_attn_output = attention_outputs
                    ip_attn_output = None
                else:
                    attn_output, context_attn_output, ip_attn_output = attention_outputs
                    raise NotImplementedError("TaylorSeer does not support Flux IP attention outputs yet.")

                self._store_and_record(layer_idx, "img_attn", attn_output)
                hidden_states = hidden_states + self._apply_gate(gate_msa, attn_output, gate_dims)
                norm_hidden_states = block.norm2(hidden_states)
                norm_hidden_states = norm_hidden_states * (
                    1 + self._unsqueeze_gate(scale_mlp, gate_dims)
                ) + self._unsqueeze_gate(shift_mlp, gate_dims)
                ff_output = block.ff(norm_hidden_states)
                self._store_and_record(layer_idx, "img_mlp", ff_output)
                hidden_states = hidden_states + self._apply_gate(gate_mlp, ff_output, gate_dims)

                self._store_and_record(layer_idx, "txt_attn", context_attn_output)
                encoder_hidden_states = encoder_hidden_states + self._apply_gate(
                    c_gate_msa, context_attn_output, gate_dims
                )
                norm_encoder_hidden_states = block.norm2_context(encoder_hidden_states)
                norm_encoder_hidden_states = norm_encoder_hidden_states * (
                    1 + self._unsqueeze_gate(c_scale_mlp, gate_dims)
                ) + self._unsqueeze_gate(c_shift_mlp, gate_dims)
                context_ff_output = block.ff_context(norm_encoder_hidden_states)
                self._store_and_record(layer_idx, "txt_mlp", context_ff_output)
                encoder_hidden_states = encoder_hidden_states + self._apply_gate(
                    c_gate_mlp, context_ff_output, gate_dims
                )
                if ip_attn_output is not None:
                    hidden_states = hidden_states + ip_attn_output

            if encoder_hidden_states.dtype == torch.float16:
                encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
            return encoder_hidden_states, hidden_states

        return self._wrap_forward(block, forward_with_taylorseer)

    def _wrap_wan_attention_block(self, layer_idx: int, block: torch.nn.Module) -> bool:
        required = ("modulation", "norm1", "self_attn", "norm3", "cross_attn", "norm2", "ffn")
        if not all(hasattr(block, name) for name in required):
            return False
        original_forward = block.forward

        @functools.wraps(original_forward)
        def forward_with_taylorseer(x, e, seq_lens, grid_sizes, freqs, context, context_lens):
            assert e.dtype == torch.float32
            with amp.autocast(device_type="cuda", dtype=torch.float32):
                mod = (block.modulation + e).chunk(6, dim=1)
            assert mod[0].dtype == torch.float32
            module_names = ("self-attention", "cross-attention", "ffn")

            if self._should_reuse_modules(layer_idx, module_names):
                y = self._reuse_and_record(layer_idx, "self-attention")
                with amp.autocast(device_type="cuda", dtype=torch.float32):
                    x = x + y * mod[2]
                y = self._reuse_and_record(layer_idx, "cross-attention")
                x = x + y
                y = self._reuse_and_record(layer_idx, "ffn")
                with amp.autocast(device_type="cuda", dtype=torch.float32):
                    x = x + y * mod[5]
                return x

            y = block.self_attn(
                block.norm1(x).float() * (1 + mod[1]) + mod[0],
                seq_lens,
                grid_sizes,
                freqs,
            )
            self._store_and_record(layer_idx, "self-attention", y)
            with amp.autocast(device_type="cuda", dtype=torch.float32):
                x = x + y * mod[2]

            y = block.cross_attn(block.norm3(x), context, context_lens)
            self._store_and_record(layer_idx, "cross-attention", y)
            x = x + y

            y = block.ffn(block.norm2(x).float() * (1 + mod[4]) + mod[3])
            self._store_and_record(layer_idx, "ffn", y)
            with amp.autocast(device_type="cuda", dtype=torch.float32):
                x = x + y * mod[5]
            return x

        return self._wrap_forward(block, forward_with_taylorseer)

    def _ensure_step_decision(self) -> None:
        step = self._current_step_value()
        state = self._branch_state()
        if state["current_step"] == step:
            return

        state["current_step"] = step
        if step < self.warmup_steps or step >= max(0, self.num_steps - self.cooldown_steps):
            state["current_type"] = "full"
            state["cache_counter"] = 0
            self._append_anchor_step(step)
            return

        first_step = step < self.first_enhance
        fresh_interval = self.fresh_threshold
        if first_step or state["cache_counter"] == fresh_interval - 1:
            state["current_type"] = "full"
            state["cache_counter"] = 0
            self._append_anchor_step(step)
        else:
            state["current_type"] = "Taylor"
            state["cache_counter"] += 1

    def _append_anchor_step(self, step: int) -> None:
        activated_steps = self._branch_state()["activated_steps"]
        if activated_steps[-1] != step:
            activated_steps.append(step)

    def _current_type_for_step(self) -> str:
        self._ensure_step_decision()
        return self._branch_state()["current_type"]

    def _current_step_value(self) -> int:
        task = DiffusionBackend.generator.current_task
        return int(task.buffer.current_step)

    def _branch_key(self) -> str:
        cfg_branch = "pos" if DiffusionBackend.cfg_type == CFGType.POS else "neg"
        try:
            cp_group = get_cp_group()
            cp_rank = cp_group.rank_in_group if cp_group.group_size > 1 else 0
        except AssertionError:
            cp_rank = 0
        return f"{cfg_branch}_cp{cp_rank}"

    def _branch_state(self) -> Dict[str, Any]:
        return self._branch_states.setdefault(
            self._branch_key(),
            {
                "cache_counter": 0,
                "current_step": None,
                "current_type": "full",
                "activated_steps": [0],
            },
        )

    def _cache_key(self, layer: int, module: str, **kwargs):
        return ("taylorseer", self._branch_key(), int(layer), str(module))

    def _distance_from_anchor(self) -> int:
        return max(0, self._current_step_value() - self._branch_state()["activated_steps"][-1])

    def _has_module_cache(self, layer_idx: int, module_name: str) -> bool:
        return self._cache_key(layer=layer_idx, module=module_name) in DiffusionBackend.flexcache.cache

    @staticmethod
    def _flux2_split_modulation(mod: torch.Tensor, mod_param_sets: int):
        if mod.ndim == 2:
            mod = mod.unsqueeze(1)
        mod_params = torch.chunk(mod, 3 * mod_param_sets, dim=-1)
        return tuple(mod_params[3 * i : 3 * (i + 1)] for i in range(mod_param_sets))

    @staticmethod
    def _unsqueeze_gate(value: torch.Tensor, dim: Optional[int]) -> torch.Tensor:
        return value if dim is None else value.unsqueeze(dim)

    def _apply_gate(self, gate: torch.Tensor, feature: torch.Tensor, dim: Optional[int]) -> torch.Tensor:
        return self._unsqueeze_gate(gate, dim) * feature

    def _record_compute(self, decision: str, layer_idx: int, module_name: str) -> None:
        DiffusionBackend.flexcache.record_compute(
            baseline_units=1.0,
            actual_units=0.0 if decision == "reuse" else 1.0,
            task_id=getattr(DiffusionBackend.generator.current_task, "task_id", None),
            scope="taylorseer_module",
            unit="module_output",
            extra={
                "decision": decision,
                "step": self._current_step_value(),
                "branch": self._branch_key(),
                "layer": layer_idx,
                "module": module_name,
            },
        )

    def _record_step_policy(self, step: int, code: int):
        if DiffusionBackend.cfg_type != CFGType.POS:
            return
        self._vis_records[step] = max(self._vis_records.get(step, 0), code)
        self._vis_max_step = max(self._vis_max_step, step)

    def _save_policy_ppm(self, output_dir: str):
        if self._vis_max_step < 0:
            return
        os.makedirs(output_dir, exist_ok=True)
        cell = 12
        width = (self._vis_max_step + 1) * cell
        height = cell
        rgb = bytearray(width * height * 3)
        for step in range(self._vis_max_step + 1):
            code = self._vis_records.get(step, 0)
            if code == 2:
                color = (255, 180, 40)
            elif code == 1:
                color = (40, 140, 255)
            else:
                color = (160, 160, 160)
            for yy in range(height):
                for xx in range(step * cell, (step + 1) * cell):
                    idx = (yy * width + xx) * 3
                    rgb[idx] = color[0]
                    rgb[idx + 1] = color[1]
                    rgb[idx + 2] = color[2]
        path = os.path.join(output_dir, "flexcache_taylorseer_policy.ppm")
        with open(path, "wb") as f:
            f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
            f.write(rgb)
