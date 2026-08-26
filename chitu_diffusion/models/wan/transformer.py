from __future__ import annotations

from typing import Any

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel

from ...parallel.cp import EpeParallelContext
from .attention import WanCpAttnProcessor


class EpeWanTransformer3DModel(WanTransformer3DModel):
    """Official Diffusers Wan transformer with dynamic video-token CP."""

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
        processor = WanCpAttnProcessor(parallel, mode=attention_mode)
        for block in self.blocks:
            block.attn1.set_processor(processor)
        self.epe_parallel = parallel
        self._epe_attn_processor = processor

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
    ):
        parallel = getattr(self, "epe_parallel", None)
        if parallel is None or parallel.active.width == 1:
            return super().forward(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_image=encoder_hidden_states_image,
                return_dict=return_dict,
                attention_kwargs=attention_kwargs,
            )
        if encoder_hidden_states_image is not None:
            raise NotImplementedError("Wan2.1 I2V is outside the initial EPE scope")
        if attention_kwargs:
            raise NotImplementedError("Wan EPE does not support attention kwargs")
        return self._cp_forward(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=return_dict,
        )

    def _cp_forward(
        self,
        *,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        return_dict: bool,
    ):
        parallel = self.epe_parallel
        topology = parallel.active
        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_frames = num_frames // p_t
        post_height = height // p_h
        post_width = width // p_w

        rotary_emb = self.rope(hidden_states)
        hidden_states = self.patch_embedding(hidden_states).flatten(2).transpose(1, 2)
        image_tokens = hidden_states.shape[1]
        if image_tokens % topology.width:
            raise ValueError(
                f"Wan video token count {image_tokens} is not divisible by lane width {topology.width}"
            )
        local_tokens = image_tokens // topology.width
        start = topology.rank_in_lane * local_tokens
        stop = start + local_tokens
        hidden_states = hidden_states[:, start:stop].contiguous()
        rotary_emb = tuple(value[:, start:stop].contiguous() for value in rotary_emb)

        temb, timestep_proj, encoder_hidden_states, _ = self.condition_embedder(
            timestep, encoder_hidden_states, None, timestep_seq_len=None
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                rotary_emb,
            )

        shift, scale = (
            self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)
        ).chunk(2, dim=1)
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)
        hidden_states = (
            self.norm_out(hidden_states.float()) * (1 + scale) + shift
        ).type_as(hidden_states)
        hidden_states = parallel.all_gather_sequence(self.proj_out(hidden_states))
        hidden_states = hidden_states.reshape(
            batch_size,
            post_frames,
            post_height,
            post_width,
            p_t,
            p_h,
            p_w,
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
