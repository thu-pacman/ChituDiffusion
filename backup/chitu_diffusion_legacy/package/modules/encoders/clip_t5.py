from __future__ import annotations

from typing import Optional

import torch
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast


class CLIPT5TextEncoder:
    """CLIP pooled embeddings plus T5 sequence embeddings."""

    def __init__(self, model_path: str, device: torch.device, dtype: torch.dtype = torch.bfloat16):
        self.model_path = model_path
        self.device = torch.device(device)
        self.dtype = dtype
        self.tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        self.tokenizer_2 = T5TokenizerFast.from_pretrained(model_path, subfolder="tokenizer_2")
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_path,
            subfolder="text_encoder",
            torch_dtype=dtype,
        ).to(device)
        self.text_encoder_2 = T5EncoderModel.from_pretrained(
            model_path,
            subfolder="text_encoder_2",
            torch_dtype=dtype,
        ).to(device)
        self.text_encoder.eval().requires_grad_(False)
        self.text_encoder_2.eval().requires_grad_(False)
        self.tokenizer_max_length = self.tokenizer.model_max_length

    @property
    def model(self):
        return self

    def to(self, device):
        self.device = torch.device(device)
        self.text_encoder.to(device)
        self.text_encoder_2.to(device)
        return self

    def parameters(self):
        yield from self.text_encoder.parameters()
        yield from self.text_encoder_2.parameters()

    @torch.no_grad()
    def encode(
        self,
        prompt: str,
        *,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if max_sequence_length > 512:
            raise ValueError(f"CLIP+T5 max_sequence_length cannot exceed 512, got {max_sequence_length}.")
        device = torch.device(device or self.device)
        prompt_list = [prompt]

        clip_inputs = self.tokenizer(
            prompt_list,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )
        pooled = self.text_encoder(
            clip_inputs.input_ids.to(device),
            output_hidden_states=False,
        ).pooler_output
        pooled = pooled.to(dtype=self.text_encoder.dtype, device=device)

        t5_inputs = self.tokenizer_2(
            prompt_list,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        prompt_embeds = self.text_encoder_2(
            t5_inputs.input_ids.to(device),
            output_hidden_states=False,
        )[0]
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
        text_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=pooled.dtype)
        return prompt_embeds, pooled, text_ids
