from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ...parallel.cp import EpeParallelContext
from .attention import LLaDAImageCpAttnProcessor
from .diffusers_components import LLaDAImageTransformer2DModel
from .epe import LLaDAImageEpeModule


@dataclass(slots=True)
class _CpForwardState:
    is_editing: bool
    local_image_tokens: int = 0


class EpeLLaDAImageTransformer2DModel(LLaDAImageTransformer2DModel):
    """Official LLaDA-Image transformer with Chitu image context parallelism."""

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any):
        epe_module = kwargs.pop("epe_module", None)
        parallel_context = kwargs.pop("parallel_context", None)
        model = super().from_pretrained(*args, **kwargs)
        if epe_module is not None:
            model.configure_epe(epe_module)
        elif parallel_context is not None:
            model.configure_epe(parallel_context)
        return model

    def configure_epe(
        self,
        epe: LLaDAImageEpeModule | EpeParallelContext,
        **kwargs: Any,
    ) -> None:
        if isinstance(epe, EpeParallelContext):
            epe = LLaDAImageEpeModule(epe, **kwargs)

        processors = []
        for block in [*self.noise_refiner, *self.layers]:
            native_processor = block.attention.processor
            if isinstance(native_processor, LLaDAImageCpAttnProcessor):
                native_processor = native_processor.native_processor
            processor = LLaDAImageCpAttnProcessor(
                epe.parallel,
                mode=epe.attention_mode,
                native_processor=native_processor,
            )
            block.attention.set_processor(processor)
            processors.append(processor)

        self.epe = epe
        self._epe_attn_processors = processors
        self._epe_forward_state: _CpForwardState | None = None

    def _disable_cp_processors(self) -> None:
        for processor in getattr(self, "_epe_attn_processors", ()):
            processor.disable()

    def _native_forward(self, *args: Any, **kwargs: Any):
        self._disable_cp_processors()
        try:
            return super().forward(*args, **kwargs)
        finally:
            self._logical_glm_cap_lengths = None

    def forward(
        self,
        x: list[torch.Tensor],
        t: torch.Tensor,
        cap_feats: list[torch.Tensor] | None,
        glm_cap_feats: list[torch.Tensor] | None = None,
        source_latents: list[torch.Tensor] | None = None,
        patch_size: int = 1,
        f_patch_size: int = 1,
        return_dict: bool = True,
        glm_cap_lengths: list[int] | None = None,
    ):
        epe = getattr(self, "epe", None)
        self._logical_glm_cap_lengths = glm_cap_lengths
        if epe is None or epe.parallel.active.width == 1:
            return self._native_forward(
                x,
                t,
                cap_feats,
                glm_cap_feats=glm_cap_feats,
                source_latents=source_latents,
                patch_size=patch_size,
                f_patch_size=f_patch_size,
                return_dict=return_dict,
            )
        if len(x) != 1:
            self._logical_glm_cap_lengths = None
            raise NotImplementedError(
                "LLaDA-Image context parallelism currently supports batch_size=1"
            )
        if source_latents is not None and len(source_latents) != 1:
            self._logical_glm_cap_lengths = None
            raise NotImplementedError(
                "LLaDA-Image editing context parallelism requires one source image"
            )

        self._epe_forward_state = _CpForwardState(is_editing=source_latents is not None)
        try:
            return super().forward(
                x,
                t,
                cap_feats,
                glm_cap_feats=glm_cap_feats,
                source_latents=source_latents,
                patch_size=patch_size,
                f_patch_size=f_patch_size,
                return_dict=return_dict,
            )
        finally:
            self._disable_cp_processors()
            self._epe_forward_state = None
            self._logical_glm_cap_lengths = None

    def _shard_image_sequence(self, sequence: Any) -> Any:
        state = getattr(self, "_epe_forward_state", None)
        if state is None:
            return sequence
        if len(sequence.features) != 1:
            raise NotImplementedError(
                "LLaDA-Image context parallelism does not support outer ragged batches"
            )

        topology = self.epe.parallel.active
        image_tokens = len(sequence.features[0])
        if image_tokens % topology.width:
            raise ValueError(
                f"image token count {image_tokens} is not divisible by lane width "
                f"{topology.width}"
            )
        local_tokens = image_tokens // topology.width
        start = topology.rank_in_lane * local_tokens
        stop = start + local_tokens
        sequence.features[0] = sequence.features[0][start:stop].contiguous()
        sequence.position_ids[0] = sequence.position_ids[0][start:stop].contiguous()
        sequence.padding_masks[0] = sequence.padding_masks[0][start:stop].contiguous()
        if sequence.noise_masks is not None:
            sequence.noise_masks[0] = sequence.noise_masks[0][start:stop]

        state.local_image_tokens = local_tokens
        for processor in self._epe_attn_processors:
            processor.enable(local_tokens)
        return sequence

    def _prepare_t2i_sequences(self, *args: Any, **kwargs: Any):
        result = super()._prepare_t2i_sequences(*args, **kwargs)
        if getattr(self, "_epe_forward_state", None) is None:
            return result
        image_sequence, cap_sequence, glm_sequence, image_sizes = result
        return (
            self._shard_image_sequence(image_sequence),
            cap_sequence,
            glm_sequence,
            image_sizes,
        )

    def _prepare_editing_sequences(self, *args: Any, **kwargs: Any):
        result = super()._prepare_editing_sequences(*args, **kwargs)
        if getattr(self, "_epe_forward_state", None) is None:
            return result
        image_sequence, cap_sequence, sigvq_sequence, image_sizes, offsets = result
        return (
            self._shard_image_sequence(image_sequence),
            cap_sequence,
            sigvq_sequence,
            image_sizes,
            offsets,
        )

    def _merge_padded_sequences(
        self,
        feature_groups: tuple[torch.Tensor, ...],
        frequency_groups: tuple[torch.Tensor, ...],
        length_groups: tuple[list[int], ...],
        noise_mask_groups: tuple[torch.Tensor, ...] | None = None,
    ):
        state = getattr(self, "_epe_forward_state", None)
        if (
            state is not None
            and state.is_editing
            and noise_mask_groups is not None
            and len(feature_groups) == 3
        ):
            order = (1, 0, 2)
            feature_groups = tuple(feature_groups[index] for index in order)
            frequency_groups = tuple(frequency_groups[index] for index in order)
            length_groups = tuple(length_groups[index] for index in order)
            noise_mask_groups = tuple(noise_mask_groups[index] for index in order)
        glm_cap_lengths = getattr(self, "_logical_glm_cap_lengths", None)
        if glm_cap_lengths is not None:
            if noise_mask_groups is None or len(length_groups) != 3:
                raise RuntimeError(
                    "logical GLM/SigVQ lengths are only valid for image editing"
                )
            length_groups = (*length_groups[:2], glm_cap_lengths)
        return super()._merge_padded_sequences(
            feature_groups,
            frequency_groups,
            length_groups,
            noise_mask_groups,
        )

    def _unpatchify(
        self,
        hidden_states: list[torch.Tensor],
        sizes: list[tuple[int, int, int] | list[tuple[int, int, int]]],
        patch_size: int,
        f_patch_size: int,
        image_offsets: list[tuple[int, int]] | None = None,
    ) -> list[torch.Tensor]:
        state = getattr(self, "_epe_forward_state", None)
        if state is None:
            return super()._unpatchify(
                hidden_states,
                sizes,
                patch_size,
                f_patch_size,
                image_offsets,
            )
        if len(hidden_states) != 1 or state.local_image_tokens <= 0:
            raise RuntimeError("invalid LLaDA-Image context-parallel output state")

        local_image = hidden_states[0][: state.local_image_tokens]
        full_image = self.epe.parallel.all_gather_sequence(
            local_image.unsqueeze(0).contiguous()
        )[0]
        if state.is_editing:
            return super()._unpatchify(
                [full_image],
                sizes,
                patch_size,
                f_patch_size,
                [(0, len(full_image))],
            )
        return super()._unpatchify(
            [full_image],
            sizes,
            patch_size,
            f_patch_size,
            None,
        )
