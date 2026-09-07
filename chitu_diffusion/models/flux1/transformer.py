from __future__ import annotations

from typing import Any

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel

from ...parallel.cp import EpeParallelContext
from .attention import Flux1CpAttnProcessor


class EpeFlux1Transformer2DModel(FluxTransformer2DModel):
    """Diffusers Flux transformer with dynamic image context parallelism."""

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any):
        parallel = kwargs.pop("parallel_context", None)
        attention_mode = kwargs.pop("attention_mode", "agkv")
        model = super().from_pretrained(*args, **kwargs)
        if parallel is not None:
            model.configure_epe(parallel, attention_mode=attention_mode)
        return model

    def configure_epe(
        self,
        parallel: EpeParallelContext,
        *,
        attention_mode: str = "agkv",
    ) -> None:
        processor = Flux1CpAttnProcessor(parallel, mode=attention_mode)
        for block in [*self.transformer_blocks, *self.single_transformer_blocks]:
            block.attn.set_processor(processor)
        self.epe_parallel = parallel
        self._epe_attn_processor = processor

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ):
        parallel = getattr(self, "epe_parallel", None)
        if parallel is None or parallel.active.width == 1:
            return super().forward(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance,
                joint_attention_kwargs=joint_attention_kwargs,
                controlnet_block_samples=controlnet_block_samples,
                controlnet_single_block_samples=controlnet_single_block_samples,
                return_dict=return_dict,
                controlnet_blocks_repeat=controlnet_blocks_repeat,
            )
        if joint_attention_kwargs:
            raise NotImplementedError(
                "Flux.1 CP does not support joint attention kwargs"
            )
        if (
            controlnet_block_samples is not None
            or controlnet_single_block_samples is not None
        ):
            raise NotImplementedError("Flux.1 CP does not support ControlNet")
        return self._cp_forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            return_dict=return_dict,
        )

    def _cp_forward(
        self,
        *,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        timestep: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        guidance: torch.Tensor | None,
        return_dict: bool,
    ):
        parallel = self.epe_parallel
        topology = parallel.active
        processor = self._epe_attn_processor
        if hidden_states.shape[1] % topology.width:
            raise ValueError(
                f"image token count {hidden_states.shape[1]} is not divisible by lane width {topology.width}"
            )
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        local_tokens = hidden_states.shape[1] // topology.width
        start = topology.rank_in_lane * local_tokens
        stop = start + local_tokens
        hidden_states = hidden_states[:, start:stop].contiguous()
        img_ids = img_ids[start:stop].contiguous()

        hidden_states = self.x_embedder(hidden_states)
        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        image_rotary_emb = self.pos_embed(torch.cat((txt_ids, img_ids), dim=0))

        processor.enable(encoder_hidden_states.shape[1])
        try:
            for block in self.transformer_blocks:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=None,
                )
            for block in self.single_transformer_blocks:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=None,
                )
        finally:
            processor.disable()

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)
        output = parallel.all_gather_sequence(output)
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
