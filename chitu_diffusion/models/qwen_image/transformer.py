from __future__ import annotations

from typing import Any

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_qwenimage import (
    QwenImageTransformer2DModel,
    compute_text_seq_len_from_mask,
)

from ...parallel import EpeParallelContext
from .attention import QwenImageCpAttnProcessor


class EpeQwenImageTransformer2DModel(QwenImageTransformer2DModel):
    """Diffusers Qwen-Image transformer with dynamic image CP."""

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any):
        parallel = kwargs.pop("parallel_context", None)
        attention_mode = kwargs.pop("attention_mode", "agkv")
        model = super().from_pretrained(*args, **kwargs)
        if parallel is not None:
            model.configure_epac(parallel, attention_mode=attention_mode)
        return model

    def configure_epac(
        self,
        parallel: EpeParallelContext,
        *,
        attention_mode: str = "agkv",
    ) -> None:
        processor = QwenImageCpAttnProcessor(parallel, mode=attention_mode)
        for block in self.transformer_blocks:
            block.attn.set_processor(processor)
        self.epac_parallel = parallel
        self._epac_attn_processor = processor

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes=None,
        txt_seq_lens=None,
        guidance: torch.Tensor = None,
        attention_kwargs: dict[str, Any] | None = None,
        controlnet_block_samples=None,
        additional_t_cond=None,
        return_dict: bool = True,
    ):
        parallel = getattr(self, "epac_parallel", None)
        if parallel is None or parallel.active.width == 1:
            return super().forward(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                timestep=timestep,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                guidance=guidance,
                attention_kwargs=attention_kwargs,
                controlnet_block_samples=controlnet_block_samples,
                additional_t_cond=additional_t_cond,
                return_dict=return_dict,
            )
        if self.zero_cond_t:
            raise NotImplementedError("Qwen-Image EPAC does not support zero_cond_t")
        if attention_kwargs:
            raise NotImplementedError(
                "Qwen-Image EPAC does not support attention kwargs"
            )
        if controlnet_block_samples is not None:
            raise NotImplementedError("Qwen-Image EPAC does not support ControlNet")
        if txt_seq_lens is not None:
            raise NotImplementedError("Qwen-Image EPAC uses encoder_hidden_states_mask")
        return self._cp_forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            timestep=timestep,
            img_shapes=img_shapes,
            guidance=guidance,
            additional_t_cond=additional_t_cond,
            return_dict=return_dict,
        )

    def _cp_forward(
        self,
        *,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor | None,
        timestep: torch.Tensor,
        img_shapes,
        guidance: torch.Tensor | None,
        additional_t_cond,
        return_dict: bool,
    ):
        parallel = self.epac_parallel
        topology = parallel.active
        image_tokens = hidden_states.shape[1]
        if image_tokens % topology.width:
            raise ValueError(
                f"image token count {image_tokens} is not divisible by lane width {topology.width}"
            )
        if encoder_hidden_states_mask is not None:
            if not bool(encoder_hidden_states_mask.to(torch.bool).all()):
                raise NotImplementedError(
                    "Qwen-Image EPAC currently supports batch-one all-valid text only"
                )
            encoder_hidden_states_mask = None

        local_tokens = image_tokens // topology.width
        start = topology.rank_in_lane * local_tokens
        stop = start + local_tokens
        hidden_states = hidden_states[:, start:stop].contiguous()
        hidden_states = self.img_in(hidden_states)
        timestep = timestep.to(hidden_states.dtype)
        encoder_hidden_states = self.txt_in(self.txt_norm(encoder_hidden_states))
        text_seq_len, _, _ = compute_text_seq_len_from_mask(
            encoder_hidden_states, encoder_hidden_states_mask
        )
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        temb = (
            self.time_text_embed(timestep, hidden_states, additional_t_cond)
            if guidance is None
            else self.time_text_embed(
                timestep, guidance, hidden_states, additional_t_cond
            )
        )
        image_freqs, text_freqs = self.pos_embed(
            img_shapes,
            max_txt_seq_len=text_seq_len,
            device=hidden_states.device,
        )
        image_rotary_emb = (image_freqs[start:stop].contiguous(), text_freqs)

        processor = self._epac_attn_processor
        processor.enable()
        try:
            for block in self.transformer_blocks:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=None,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=None,
                    modulate_index=None,
                )
        finally:
            processor.disable()

        hidden_states = self.norm_out(hidden_states, temb)
        output = parallel.all_gather_sequence(self.proj_out(hidden_states))
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
