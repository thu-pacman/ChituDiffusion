from __future__ import annotations

from typing import Any

import torch

from chitu_diffusion.flexcache.config import TaylorSeerConfig

from ..contracts import CacheStepContext
from ..spec import BlockSite, FlexCacheModelSpec, LeafSite
from ..tree import (
    TensorTree,
    tree_add,
    tree_clone,
    tree_mul,
    tree_nbytes,
    tree_sub,
)
from .base import BaseCacheStrategy


class TaylorSeerStrategy(BaseCacheStrategy):
    """Extrapolate pre-gate submodule outputs with a finite-difference series.

    Generic models reuse leaf outputs. Flux.1 additionally has a block hook
    fast path: on predicted steps it evaluates only the current modulation
    gates and combines them with predicted attention/MLP outputs. This skips
    the same expensive block work as the reference implementation without
    changing Diffusers model code.
    """

    def __init__(
        self,
        params: TaylorSeerConfig,
        *,
        warmup_steps: int,
        cooldown_steps: int,
    ) -> None:
        super().__init__()
        self.params = params
        self.warmup_steps = warmup_steps
        self.cooldown_steps = cooldown_steps
        self.reuse_current_step = False
        self.anchor_steps: list[int] = []
        self.factors: dict[str, list[TensorTree]] = {}
        self.selected: frozenset[str] = frozenset()
        self.leaves_by_block: dict[int, dict[str, LeafSite]] = {}

    def begin(self, *, total_steps: int, model_spec: FlexCacheModelSpec) -> None:
        super().begin(total_steps=total_steps, model_spec=model_spec)
        self.reuse_current_step = False
        self.anchor_steps.clear()
        self.factors.clear()
        self.selected = self._select_leaves(model_spec)
        self.leaves_by_block = self._index_leaves(model_spec)
        if not self.selected:
            raise ValueError(
                f"{model_spec.family} exposes no pre-gate submodules for TaylorSeer"
            )
        self.stats.strategy = {
            "fresh_threshold": self.params.fresh_threshold,
            "max_order": self.params.max_order,
            "first_enhance": self.params.first_enhance,
            "leaf_sites": len(self.selected),
            "flux_block_fast_path": model_spec.family == "flux1",
        }

    @staticmethod
    def _select_leaves(model_spec: FlexCacheModelSpec) -> frozenset[str]:
        """Prefer a block's terminal projection over its constituent parts."""

        by_block: dict[int, list[LeafSite]] = {}
        for leaf in model_spec.leaves:
            by_block.setdefault(leaf.block_index, []).append(leaf)
        selected: set[str] = set()
        for leaves in by_block.values():
            projections = [leaf for leaf in leaves if leaf.kind == "projection"]
            chosen = projections or leaves
            selected.update(leaf.site_id for leaf in chosen)
        return frozenset(selected)

    @staticmethod
    def _index_leaves(
        model_spec: FlexCacheModelSpec,
    ) -> dict[int, dict[str, LeafSite]]:
        indexed: dict[int, dict[str, LeafSite]] = {}
        for leaf in model_spec.leaves:
            indexed.setdefault(leaf.block_index, {})[
                leaf.site_id.rsplit(".", 1)[-1]
            ] = leaf
        return indexed

    def _anchor_distance(self) -> int:
        if len(self.anchor_steps) < 2:
            return 1
        return max(1, self.anchor_steps[-1] - self.anchor_steps[-2])

    def model_lookup(
        self,
        context: CacheStepContext,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[bool, TensorTree | None]:
        del args, kwargs
        force = (
            context.step_index < max(self.warmup_steps, self.params.first_enhance)
            or context.step_index >= self.total_steps - self.cooldown_steps
            or not self.anchor_steps
            or context.step_index - self.anchor_steps[-1] >= self.params.fresh_threshold
        )
        if force:
            self.reuse_current_step = False
            self.anchor_steps.append(context.step_index)
            keep = max(2, self.params.max_order + 1)
            if len(self.anchor_steps) > keep:
                del self.anchor_steps[:-keep]
            self.stats.step_misses += 1
            return False, None
        self.reuse_current_step = True
        self.stats.step_hits += 1
        return False, None

    def block_lookup(
        self,
        context: CacheStepContext,
        site: BlockSite,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[bool, TensorTree | None]:
        if (
            not self.reuse_current_step
            or context.model_spec.family != "flux1"
            or not self.factors
        ):
            return False, None
        output = self._flux_block_prediction(context, site, args, kwargs)
        if output is None:
            return False, None
        selected = self.leaves_by_block.get(site.index, {})
        self.stats.block_hits += 1
        self.stats.leaf_hits += sum(
            leaf.site_id in self.selected for leaf in selected.values()
        )
        return True, output

    def _flux_block_prediction(
        self,
        context: CacheStepContext,
        site: BlockSite,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> TensorTree | None:
        if site.group_name == "transformer_blocks":
            return self._flux_double_prediction(context, site, args, kwargs)
        if site.group_name == "single_transformer_blocks":
            return self._flux_single_prediction(context, site, args, kwargs)
        return None

    @staticmethod
    def _argument(
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        name: str,
        position: int,
    ) -> Any:
        return kwargs.get(name, args[position] if len(args) > position else None)

    def _predicted_leaf(
        self,
        context: CacheStepContext,
        site: BlockSite,
        name: str,
    ) -> TensorTree | None:
        leaf = self.leaves_by_block.get(site.index, {}).get(name)
        if leaf is None:
            return None
        factors = self.factors.get(leaf.site_id)
        if not factors:
            return None
        distance = context.step_index - self.anchor_steps[-1]
        return self._predict(factors, distance)

    def _flux_double_prediction(
        self,
        context: CacheStepContext,
        site: BlockSite,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        hidden_states = self._argument(args, kwargs, "hidden_states", 0)
        encoder_hidden_states = self._argument(args, kwargs, "encoder_hidden_states", 1)
        temb = self._argument(args, kwargs, "temb", 2)
        attention_outputs = self._predicted_leaf(context, site, "attn")
        ff_output = self._predicted_leaf(context, site, "ff")
        context_ff_output = self._predicted_leaf(context, site, "ff_context")
        if (
            not isinstance(hidden_states, torch.Tensor)
            or not isinstance(encoder_hidden_states, torch.Tensor)
            or not isinstance(temb, torch.Tensor)
            or not isinstance(attention_outputs, (tuple, list))
            or len(attention_outputs) not in {2, 3}
            or not isinstance(ff_output, torch.Tensor)
            or not isinstance(context_ff_output, torch.Tensor)
        ):
            return None
        attn_output, context_attn_output = attention_outputs[:2]
        if not isinstance(attn_output, torch.Tensor) or not isinstance(
            context_attn_output, torch.Tensor
        ):
            return None

        module = site.module
        _, gate_msa, _, _, gate_mlp = module.norm1(hidden_states, emb=temb)
        _, c_gate_msa, _, _, c_gate_mlp = module.norm1_context(
            encoder_hidden_states, emb=temb
        )
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output
        if len(attention_outputs) == 3:
            ip_attn_output = attention_outputs[2]
            if not isinstance(ip_attn_output, torch.Tensor):
                return None
            hidden_states = hidden_states + ip_attn_output
        encoder_hidden_states = (
            encoder_hidden_states
            + c_gate_msa.unsqueeze(1) * context_attn_output
            + c_gate_mlp.unsqueeze(1) * context_ff_output
        )
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        return encoder_hidden_states, hidden_states

    def _flux_single_prediction(
        self,
        context: CacheStepContext,
        site: BlockSite,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        hidden_states = self._argument(args, kwargs, "hidden_states", 0)
        encoder_hidden_states = self._argument(args, kwargs, "encoder_hidden_states", 1)
        temb = self._argument(args, kwargs, "temb", 2)
        projection = self._predicted_leaf(context, site, "proj_out")
        if (
            not isinstance(hidden_states, torch.Tensor)
            or not isinstance(encoder_hidden_states, torch.Tensor)
            or not isinstance(temb, torch.Tensor)
            or not isinstance(projection, torch.Tensor)
        ):
            return None

        text_seq_len = encoder_hidden_states.shape[1]
        merged = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        _, gate = site.module.norm(merged, emb=temb)
        merged = merged + gate.unsqueeze(1) * projection
        if merged.dtype == torch.float16:
            merged = merged.clip(-65504, 65504)
        return merged[:, :text_seq_len], merged[:, text_seq_len:]

    def leaf_lookup(
        self,
        context: CacheStepContext,
        site: LeafSite,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[bool, TensorTree | None]:
        del args, kwargs
        if site.site_id not in self.selected:
            return False, None
        factors = self.factors.get(site.site_id)
        if not self.reuse_current_step or not factors:
            self.stats.leaf_misses += 1
            return False, None
        distance = context.step_index - self.anchor_steps[-1]
        self.stats.leaf_hits += 1
        return True, self._predict(factors, distance)

    @staticmethod
    def _predict(factors: list[TensorTree], distance: int) -> TensorTree:
        output = tree_clone(factors[0])
        weight = 1.0
        for order in range(1, len(factors)):
            weight *= distance / order
            output = tree_add(output, tree_mul(factors[order], weight))
        return output

    def leaf_store(
        self,
        context: CacheStepContext,
        site: LeafSite,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: TensorTree,
    ) -> None:
        del args, kwargs
        if site.site_id not in self.selected:
            return None
        if (
            context.model_spec.family == "flux1"
            and site.site_id.endswith(".attn")
            and (not isinstance(output, (tuple, list)) or len(output) != 2)
        ):
            raise NotImplementedError(
                "TaylorSeer Flux.1 requires two-output joint attention; "
                "IP-Adapter attention outputs are not supported"
            )
        previous = self.factors.get(site.site_id, [])
        distance = float(self._anchor_distance())
        updated: list[TensorTree] = [tree_clone(output)]
        if context.step_index > self.params.first_enhance - 2:
            for order in range(self.params.max_order):
                if order >= len(previous):
                    break
                updated.append(
                    tree_mul(
                        tree_sub(updated[order], previous[order]),
                        1.0 / distance,
                    )
                )
        self.factors[site.site_id] = updated
        self.stats.cache_bytes = sum(
            tree_nbytes(value) for values in self.factors.values() for value in values
        )
        return None
