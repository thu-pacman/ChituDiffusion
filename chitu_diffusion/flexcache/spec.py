from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn

LeafKind = Literal["self", "cross", "joint", "feedforward", "projection"]

ATTENTION_KINDS: frozenset[str] = frozenset({"self", "cross", "joint"})


@dataclass(frozen=True, slots=True)
class BlockSite:
    site_id: str
    index: int
    group_name: str
    group_index: int
    group_size: int
    module: nn.Module


@dataclass(frozen=True, slots=True)
class LeafSite:
    """A submodule whose raw output precedes the block's gate and residual.

    Caching here rather than at the block boundary lets a strategy reuse the
    expensive projection while the block still applies the *current* step's
    modulation, which is what the TaylorSeer reference does.
    """

    site_id: str
    block_index: int
    group_name: str
    kind: LeafKind
    module: nn.Module

    @property
    def is_attention(self) -> bool:
        return self.kind in ATTENTION_KINDS


@dataclass(frozen=True, slots=True)
class FlexCacheModelSpec:
    family: str
    model: nn.Module
    blocks: tuple[BlockSite, ...]
    leaves: tuple[LeafSite, ...]
    timestep_arg: str
    timestep_position: int

    @property
    def attentions(self) -> tuple[LeafSite, ...]:
        return tuple(leaf for leaf in self.leaves if leaf.is_attention)

    def step_probe(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if self.timestep_arg in kwargs:
            return kwargs[self.timestep_arg]
        if len(args) > self.timestep_position:
            return args[self.timestep_position]
        return None

    def block_input(
        self,
        site: BlockSite,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        del site
        hidden = self._argument(args, kwargs, "hidden_states", 0)
        if self.family in {"flux1", "flux2_klein", "qwen_image"}:
            encoder = self._argument(args, kwargs, "encoder_hidden_states", 1)
            if encoder is not None:
                return (encoder, hidden)
        return hidden

    @staticmethod
    def _argument(
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        name: str,
        position: int,
    ) -> Any:
        return kwargs.get(name, args[position] if len(args) > position else None)

    def variant(self) -> str:
        """Coarse checkpoint identity used to select calibrated constants."""

        config = getattr(self.model, "config", None)
        heads = getattr(config, "num_attention_heads", None)
        head_dim = getattr(config, "attention_head_dim", None)
        inner_dim = heads * head_dim if heads and head_dim else None
        if self.family == "wan":
            conditioning = "i2v" if getattr(config, "image_dim", None) else "t2v"
            if inner_dim == 1536:
                return f"wan-{conditioning}-1.3b"
            if inner_dim == 5120:
                return f"wan-{conditioning}-14b"
            return f"wan-{conditioning}-unknown"
        return self.family

    def modulation_probe(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> torch.Tensor | None:
        """Project model inputs to the signal whose drift TeaCache measures.

        The probe is deliberately computed from the transformer's *inputs*,
        which are full-sequence and replicated on every context-parallel rank.
        Recomputing the projection here costs one embedding pass but keeps the
        reuse decision identical across CP and CFG-parallel ranks, so a lane
        never splits into ranks that disagree about which steps to skip.
        """

        timestep = self.step_probe(args, kwargs)
        if not isinstance(timestep, torch.Tensor):
            return None
        with torch.no_grad():
            if self.family == "zimage":
                return self._zimage_probe(timestep)
            if self.family in {"flux1", "flux2_klein"}:
                return self._flux_probe(timestep, args, kwargs)
            if self.family == "qwen_image":
                return self._qwen_probe(timestep, args, kwargs)
            if self.family == "wan":
                return self._wan_probe(timestep, args, kwargs)
        return timestep

    def _zimage_probe(self, timestep: torch.Tensor) -> torch.Tensor | None:
        embedder = getattr(self.model, "t_embedder", None)
        if embedder is None:
            return None
        return embedder(timestep * self.model.t_scale)

    def _flux_probe(
        self,
        timestep: torch.Tensor,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> torch.Tensor | None:
        """Reference TeaCache probe: the first block's modulated hidden states.

        The published polynomials were fitted against this signal rather than
        against the raw time embedding, so the threshold only carries its
        documented meaning when the same signal is measured.
        """

        embedder = getattr(self.model, "time_text_embed", None)
        patchifier = getattr(self.model, "x_embedder", None)
        blocks = getattr(self.model, "transformer_blocks", None)
        pooled = self._argument(args, kwargs, "pooled_projections", 2)
        hidden = self._argument(args, kwargs, "hidden_states", 0)
        if (
            embedder is None
            or patchifier is None
            or not blocks
            or pooled is None
            or not isinstance(hidden, torch.Tensor)
        ):
            return None
        hidden = patchifier(hidden)
        scaled = timestep.to(hidden.dtype) * 1000
        guidance = self._argument(args, kwargs, "guidance", 6)
        if guidance is None:
            temb = embedder(scaled, pooled)
        else:
            temb = embedder(scaled, guidance.to(hidden.dtype) * 1000, pooled)
        modulated = blocks[0].norm1(hidden, emb=temb)
        return modulated[0] if isinstance(modulated, tuple) else modulated

    def _qwen_probe(
        self,
        timestep: torch.Tensor,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> torch.Tensor | None:
        embedder = getattr(self.model, "time_text_embed", None)
        patchifier = getattr(self.model, "img_in", None)
        hidden = self._argument(args, kwargs, "hidden_states", 0)
        if embedder is None or patchifier is None or not isinstance(hidden, torch.Tensor):
            return None
        hidden = patchifier(hidden)
        additional = self._argument(args, kwargs, "additional_t_cond", 10)
        guidance = self._argument(args, kwargs, "guidance", 6)
        scaled = timestep.to(hidden.dtype)
        if guidance is None:
            return embedder(scaled, hidden, additional)
        return embedder(scaled, guidance.to(hidden.dtype) * 1000, hidden, additional)

    def _wan_probe(
        self,
        timestep: torch.Tensor,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> torch.Tensor | None:
        """Reference TeaCache probe for Wan without retention steps: ``e``.

        Diffusers' ``condition_embedder`` returns the same tensor the reference
        calls ``e``; the retention-step variant instead probes the projected
        ``e0``, which pairs with a different polynomial.
        """

        embedder = getattr(self.model, "condition_embedder", None)
        if embedder is None:
            return None
        encoder = self._argument(args, kwargs, "encoder_hidden_states", 2)
        temb, *_ = embedder(timestep, encoder, None, timestep_seq_len=None)
        return temb

    @classmethod
    def discover(cls, model: nn.Module, *, family: str) -> "FlexCacheModelSpec":
        try:
            contract = _FAMILY_CONTRACTS[family]
        except KeyError as exc:
            raise ValueError(f"unsupported FlexCache model family: {family}") from exc

        block_groups: list[tuple[str, object]] = []
        for name in contract.block_groups:
            value = getattr(model, name, None)
            if isinstance(value, (nn.ModuleList, list, tuple)):
                block_groups.append((name, value))

        blocks: list[BlockSite] = []
        leaves: list[LeafSite] = []
        seen_blocks: set[int] = set()
        seen_leaves: set[int] = set()
        block_index = 0
        for group_name, group in block_groups:
            group_size = sum(isinstance(block, nn.Module) for block in group)
            leaf_contract = contract.leaves_for(group_name)
            for local_index, block in enumerate(group):
                if not isinstance(block, nn.Module) or id(block) in seen_blocks:
                    continue
                seen_blocks.add(id(block))
                site_id = f"{family}.block.{group_name}.{local_index}"
                blocks.append(
                    BlockSite(
                        site_id,
                        block_index,
                        group_name,
                        local_index,
                        group_size,
                        block,
                    )
                )
                for attr_name, kind in leaf_contract:
                    leaf = getattr(block, attr_name, None)
                    if not isinstance(leaf, nn.Module) or id(leaf) in seen_leaves:
                        continue
                    seen_leaves.add(id(leaf))
                    leaves.append(
                        LeafSite(
                            site_id=f"{site_id}.{attr_name}",
                            block_index=block_index,
                            group_name=group_name,
                            kind=kind,
                            module=leaf,
                        )
                    )
                block_index += 1

        if not blocks:
            raise ValueError(
                f"{model.__class__.__name__} exposes no FlexCache backbone blocks"
            )
        return cls(
            family=family,
            model=model,
            blocks=tuple(blocks),
            leaves=tuple(leaves),
            timestep_arg=contract.timestep_arg,
            timestep_position=contract.timestep_position,
        )


@dataclass(frozen=True, slots=True)
class _FamilyContract:
    block_groups: tuple[str, ...]
    timestep_arg: str
    timestep_position: int
    leaves: dict[str, tuple[tuple[str, LeafKind], ...]]

    def leaves_for(self, group_name: str) -> tuple[tuple[str, LeafKind], ...]:
        return self.leaves.get(group_name, self.leaves.get("*", ()))


_ZIMAGE_LEAVES: tuple[tuple[str, LeafKind], ...] = (
    ("attention", "joint"),
    ("feed_forward", "feedforward"),
)
_FLUX_DOUBLE_LEAVES: tuple[tuple[str, LeafKind], ...] = (
    ("attn", "joint"),
    ("ff", "feedforward"),
    ("ff_context", "feedforward"),
)
# The single stream fuses attention and MLP before ``proj_out``, so that
# projection is the block's only reusable pre-gate output.
_FLUX_SINGLE_LEAVES: tuple[tuple[str, LeafKind], ...] = (
    ("attn", "joint"),
    ("proj_out", "projection"),
)
_QWEN_LEAVES: tuple[tuple[str, LeafKind], ...] = (
    ("attn", "joint"),
    ("img_mlp", "feedforward"),
    ("txt_mlp", "feedforward"),
)
_WAN_LEAVES: tuple[tuple[str, LeafKind], ...] = (
    ("attn1", "self"),
    ("attn2", "cross"),
    ("ffn", "feedforward"),
)

_FAMILY_CONTRACTS: dict[str, _FamilyContract] = {
    "zimage": _FamilyContract(
        block_groups=("noise_refiner", "context_refiner", "layers"),
        timestep_arg="t",
        timestep_position=1,
        leaves={"*": _ZIMAGE_LEAVES},
    ),
    "flux1": _FamilyContract(
        block_groups=("transformer_blocks", "single_transformer_blocks"),
        timestep_arg="timestep",
        timestep_position=3,
        leaves={
            "transformer_blocks": _FLUX_DOUBLE_LEAVES,
            "single_transformer_blocks": _FLUX_SINGLE_LEAVES,
        },
    ),
    "flux2_klein": _FamilyContract(
        block_groups=("transformer_blocks", "single_transformer_blocks"),
        timestep_arg="timestep",
        timestep_position=3,
        leaves={
            "transformer_blocks": _FLUX_DOUBLE_LEAVES,
            "single_transformer_blocks": _FLUX_SINGLE_LEAVES,
        },
    ),
    "qwen_image": _FamilyContract(
        block_groups=("transformer_blocks",),
        timestep_arg="timestep",
        timestep_position=3,
        leaves={"*": _QWEN_LEAVES},
    ),
    "wan": _FamilyContract(
        block_groups=("blocks",),
        timestep_arg="timestep",
        timestep_position=1,
        leaves={"*": _WAN_LEAVES},
    ),
}
