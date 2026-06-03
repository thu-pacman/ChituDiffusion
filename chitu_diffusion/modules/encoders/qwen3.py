from typing import Iterable, Optional

import torch
from transformers import (
    Qwen2TokenizerFast,
    Qwen3ForCausalLM,
)

OUTPUT_LAYERS_QWEN3 = [9, 18, 27]


class Qwen3CausalLMTextEncoder:
    """Qwen3 hidden-state text encoder built from selected causal-LM layers."""

    def __init__(self, model_path: str, device: torch.device, dtype: torch.dtype = torch.bfloat16):
        self.model_path = model_path
        self.device = torch.device(device)
        self.dtype = dtype
        self.tokenizer = Qwen2TokenizerFast.from_pretrained(model_path, subfolder="tokenizer")
        self.text_encoder = Qwen3ForCausalLM.from_pretrained(
            model_path,
            subfolder="text_encoder",
            torch_dtype=dtype,
        ).to(device)
        self.text_encoder.eval().requires_grad_(False)

    @property
    def model(self):
        return self

    def to(self, device):
        self.device = torch.device(device)
        self.text_encoder.to(device)
        return self

    def parameters(self):
        yield from self.text_encoder.parameters()

    @torch.no_grad()
    def encode(
        self,
        prompt: str,
        *,
        max_sequence_length: int = 512,
        hidden_states_layers: Iterable[int] = OUTPUT_LAYERS_QWEN3,
        device: Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = torch.device(device or self.device)
        messages = [{"role": "user", "content": prompt or ""}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
        )
        output = self.text_encoder(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            output_hidden_states=True,
            use_cache=False,
        )
        states = torch.stack([output.hidden_states[int(k)] for k in hidden_states_layers], dim=1)
        states = states.to(dtype=self.text_encoder.dtype, device=device)
        batch_size, num_channels, seq_len, hidden_dim = states.shape
        prompt_embeds = states.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)
        text_ids = prepare_qwen3_text_ids(prompt_embeds).to(device)
        return prompt_embeds, text_ids


def prepare_qwen3_text_ids(x: torch.Tensor, t_coord: torch.Tensor | None = None) -> torch.Tensor:
    batch_size, seq_len, _ = x.shape
    out_ids = []
    for i in range(batch_size):
        t = torch.arange(1) if t_coord is None else t_coord[i]
        h = torch.arange(1)
        w = torch.arange(1)
        l = torch.arange(seq_len)
        out_ids.append(torch.cartesian_prod(t, h, w, l))
    return torch.stack(out_ids)
