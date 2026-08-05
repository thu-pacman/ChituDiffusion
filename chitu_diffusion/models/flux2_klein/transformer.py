from __future__ import annotations

from typing import Any

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel

from ...parallel import EpeParallelContext, resolve_context_parallel_config
from .attention import (
    Flux2KleinCpAttnProcessor,
    Flux2KleinCpSingleAttnProcessor,
)


class Flux2KleinCpTransformer2DModel(Flux2Transformer2DModel):
    """Diffusers FLUX.2 transformer with image context parallelism."""

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any):
        parallel = kwargs.pop("parallel_context", None)
        attention_mode, _ = resolve_context_parallel_config(
            kwargs.pop("attention_mode", "agkv")
        )
        model = super().from_pretrained(*args, **kwargs)
        if parallel is not None:
            model.configure_context_parallel(parallel, attention_mode=attention_mode)
        return model

    def configure_context_parallel(
        self,
        parallel: EpeParallelContext,
        *,
        attention_mode: str = "agkv",
    ) -> None:
        double_processor = Flux2KleinCpAttnProcessor(parallel, mode=attention_mode)
        single_processor = Flux2KleinCpSingleAttnProcessor(
            parallel, mode=attention_mode
        )
        for block in self.transformer_blocks:
            block.attn.set_processor(double_processor)
        for block in self.single_transformer_blocks:
            block.attn.set_processor(single_processor)
        self.cp_parallel = parallel
        self._cp_double_processor = double_processor
        self._cp_single_processor = single_processor

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ):
        parallel = getattr(self, "cp_parallel", None)
        if parallel is None or parallel.active.width == 1:
            return super().forward(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance,
                joint_attention_kwargs=joint_attention_kwargs,
                return_dict=return_dict,
            )
        if joint_attention_kwargs:
            raise NotImplementedError(
                "FLUX.2-klein CP does not support joint attention kwargs"
            )
        return self._cp_forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
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
        timestep: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        guidance: torch.Tensor | None,
        return_dict: bool,
    ):
        parallel = self.cp_parallel
        topology = parallel.active
        if hidden_states.shape[1] % topology.width:
            raise ValueError(
                f"image token count {hidden_states.shape[1]} is not divisible "
                f"by lane width {topology.width}"
            )
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        local_tokens = hidden_states.shape[1] // topology.width
        start = topology.rank_in_lane * local_tokens
        stop = start + local_tokens
        hidden_states = hidden_states[:, start:stop].contiguous()
        img_ids = img_ids[start:stop].contiguous()
        text_tokens = encoder_hidden_states.shape[1]

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        temb = self.time_guidance_embed(timestep, guidance)
        double_mod_img = self.double_stream_modulation_img(temb)
        double_mod_txt = self.double_stream_modulation_txt(temb)
        single_mod = self.single_stream_modulation(temb)
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        image_rotary_emb = self.pos_embed(img_ids)
        text_rotary_emb = self.pos_embed(txt_ids)
        concat_rotary_emb = (
            torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
            torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
        )
        double_processor = self._cp_double_processor
        single_processor = self._cp_single_processor
        double_processor.enable(text_tokens)
        single_processor.enable(text_tokens)
        try:
            for block in self.transformer_blocks:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb_mod_img=double_mod_img,
                    temb_mod_txt=double_mod_txt,
                    image_rotary_emb=concat_rotary_emb,
                    joint_attention_kwargs=None,
                )
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            for block in self.single_transformer_blocks:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=None,
                    temb_mod=single_mod,
                    image_rotary_emb=concat_rotary_emb,
                    joint_attention_kwargs=None,
                )
        finally:
            double_processor.disable()
            single_processor.disable()

        hidden_states = hidden_states[:, text_tokens:]
        hidden_states = self.norm_out(hidden_states, temb)
        output = parallel.all_gather_sequence(self.proj_out(hidden_states))
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
