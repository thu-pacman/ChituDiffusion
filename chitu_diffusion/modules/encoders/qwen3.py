from pathlib import Path

import torch
import torch.nn as nn
from einops import rearrange
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Mistral3ForConditionalGeneration,
    pipeline,
)

from einops import rearrange

OUTPUT_LAYERS_MISTRAL = [10, 20, 30]
OUTPUT_LAYERS_QWEN3 = [9, 18, 27]
MAX_LENGTH = 512
NSFW_THRESHOLD = 0.85
UPSAMPLING_MAX_IMAGE_SIZE = 768**2


class Qwen3Embedder(nn.Module):
    def __init__(
        self,
        model_spec: str,
        device: str | torch.device = "cuda",
    ):
        super().__init__()

        from transformers import AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(
            model_spec,
            torch_dtype=None,
            device_map=str(device),
        )

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_spec)
        self.max_length = MAX_LENGTH

    @torch.no_grad()
    def forward(self, txt: list[str]):
        all_input_ids = []
        all_attention_masks = []

        for prompt in txt:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            model_inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )

            all_input_ids.append(model_inputs["input_ids"])
            all_attention_masks.append(model_inputs["attention_mask"])

        input_ids = torch.cat(all_input_ids, dim=0).to(self.model.device)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(self.model.device)

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        out = torch.stack([output.hidden_states[k] for k in OUTPUT_LAYERS_QWEN3], dim=1)
        return rearrange(out, "b c l d -> b l (c d)")