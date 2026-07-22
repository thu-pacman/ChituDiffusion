from __future__ import annotations

from typing import Any

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel

from .attention import ZImageCpAttnProcessor
from .epe import ZImageEpeModule
from ...parallel import EpeParallelContext


class EpeZImageTransformer2DModel(ZImageTransformer2DModel):
    """Diffusers Z-Image transformer with topology-aware context parallelism."""

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
        self, epe: ZImageEpeModule | EpeParallelContext, **kwargs: Any
    ) -> None:
        if isinstance(epe, EpeParallelContext):
            epe = ZImageEpeModule(epe, **kwargs)
        processor = ZImageCpAttnProcessor(
            epe.parallel,
            mode=epe.attention_mode,
        )
        for block in [*self.noise_refiner, *self.context_refiner, *self.layers]:
            block.attention.set_processor(processor)
        self.epe = epe
        self._epe_attn_processor = processor

    def _native_forward(self, *args: Any, **kwargs: Any):
        return super().forward(*args, **kwargs)

    def forward(
        self,
        x,
        t,
        cap_feats,
        return_dict: bool = True,
        controlnet_block_samples=None,
        siglip_feats=None,
        image_noise_mask=None,
        patch_size: int = 2,
        f_patch_size: int = 1,
    ):
        epe = getattr(self, "epe", None)
        if epe is None or epe.parallel.active.width == 1:
            return self._native_forward(
                x,
                t,
                cap_feats,
                return_dict=return_dict,
                controlnet_block_samples=controlnet_block_samples,
                siglip_feats=siglip_feats,
                image_noise_mask=image_noise_mask,
                patch_size=patch_size,
                f_patch_size=f_patch_size,
            )

        if len(x) > 1:
            outputs = []
            for index in range(len(x)):
                branch = self.forward(
                    [x[index]],
                    t[index : index + 1],
                    [cap_feats[index]],
                    return_dict=False,
                    controlnet_block_samples=None,
                    siglip_feats=None,
                    image_noise_mask=None,
                    patch_size=patch_size,
                    f_patch_size=f_patch_size,
                )[0]
                outputs.extend(branch)
            return (
                (outputs,)
                if not return_dict
                else Transformer2DModelOutput(sample=outputs)
            )

        if (
            isinstance(x[0], list)
            or controlnet_block_samples is not None
            or siglip_feats is not None
            or image_noise_mask is not None
        ):
            raise NotImplementedError(
                "EPE Z-Image CP currently supports basic text-to-image inputs only"
            )

        return self._cp_forward(
            x=x,
            t=t,
            cap_feats=cap_feats,
            return_dict=return_dict,
            patch_size=patch_size,
            f_patch_size=f_patch_size,
        )

    def _cp_forward(
        self,
        *,
        x,
        t,
        cap_feats,
        return_dict: bool,
        patch_size: int,
        f_patch_size: int,
    ):
        epe = getattr(self, "epe", None)
        processor = getattr(self, "_epe_attn_processor", None)
        assert epe is not None and processor is not None
        parallel = epe.parallel
        topology = parallel.active
        device = x[0].device
        adaln_input = self.t_embedder(t * self.t_scale).type_as(x[0])

        (
            image_tokens,
            caption_tokens,
            image_size,
            image_pos_ids,
            caption_pos_ids,
            image_pad_mask,
            caption_pad_mask,
        ) = self.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)

        image_lengths = [len(tokens) for tokens in image_tokens]
        image_embeds = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](
            torch.cat(image_tokens, dim=0)
        )
        image_sequence, image_freqs, image_mask, _, image_noise = (
            self._prepare_sequence(
                list(image_embeds.split(image_lengths, dim=0)),
                image_pos_ids,
                image_pad_mask,
                self.x_pad_token,
                None,
                device,
            )
        )

        caption_lengths = [len(tokens) for tokens in caption_tokens]
        caption_embeds = self.cap_embedder(torch.cat(caption_tokens, dim=0))
        caption_sequence, caption_freqs, caption_mask, _, _ = self._prepare_sequence(
            list(caption_embeds.split(caption_lengths, dim=0)),
            caption_pos_ids,
            caption_pad_mask,
            self.cap_pad_token,
            None,
            device,
        )
        for layer in self.context_refiner:
            caption_sequence = layer(caption_sequence, caption_mask, caption_freqs)

        if image_mask is not None and bool(image_mask.all()):
            image_mask = None
        if caption_mask is not None and bool(caption_mask.all()):
            caption_mask = None
        if image_mask is not None or caption_mask is not None:
            raise NotImplementedError(
                "EPE Z-Image CP does not support padded sequences"
            )

        image_length = int(image_lengths[0])
        if image_length % topology.width:
            raise ValueError(
                f"image token count {image_length} is not divisible by lane width {topology.width}"
            )
        local_length = image_length // topology.width
        start = topology.rank_in_lane * local_length
        stop = start + local_length
        image_sequence = image_sequence[:, start:stop].contiguous()
        image_freqs = image_freqs[:, start:stop].contiguous()
        image_lengths = [local_length]

        processor.enable(local_length)
        try:
            for layer in self.noise_refiner:
                image_sequence = layer(
                    image_sequence,
                    None,
                    image_freqs,
                    adaln_input,
                    image_noise,
                    None,
                    None,
                )

            unified, unified_freqs, unified_mask, unified_noise = (
                self._build_unified_sequence(
                    image_sequence,
                    image_freqs,
                    image_lengths,
                    None,
                    caption_sequence,
                    caption_freqs,
                    caption_lengths,
                    None,
                    None,
                    None,
                    None,
                    None,
                    False,
                    device,
                )
            )
            for layer in self.layers:
                unified = layer(
                    unified,
                    unified_mask,
                    unified_freqs,
                    adaln_input,
                    unified_noise,
                    None,
                    None,
                )
        finally:
            processor.disable()

        projected = self.all_final_layer[f"{patch_size}-{f_patch_size}"](
            unified,
            c=adaln_input,
        )
        local_image = projected[:, :local_length].contiguous()
        full_image = parallel.all_gather_sequence(local_image)
        output = self.unpatchify(
            list(full_image.unbind(dim=0)),
            image_size,
            patch_size,
            f_patch_size,
            None,
        )
        return (output,) if not return_dict else Transformer2DModelOutput(sample=output)
