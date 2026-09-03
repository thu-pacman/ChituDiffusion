# LLaDA-Image Third-Party Notices

The following files are derived from the LLaDA-Image Diffusers implementation
in the LLaDA-Image-SGLang source tree at commit
`0ad472aa1d1f002667d59bf01ecf704673f29065`:

- `diffusers_components.py`, from
  `diffusers/src/diffusers/models/transformers/transformer_llada_image.py`.
- `pipeline.py`, from
  `diffusers/src/diffusers/pipelines/llada_image/pipeline_llada_image.py`.

The bundled LLaDA2 text encoder implementation is derived from the model code
revision `16b32c5143c95f2f` distributed with the released LLaDA-Image checkpoint:

- `text_encoder/configuration.py`
- `text_encoder/modeling.py`

Its dependency-specific MoE call is replaced by the local implementation in
`text_encoder_ops.py`; no external MoE runtime package is required.

The Diffusers source files are Copyright 2026 The HuggingFace Team. The text
encoder source files are Copyright 2025 Antgroup and The HuggingFace Inc. team.
All are distributed under the Apache License, Version 2.0. The original Apache-2.0 headers are
preserved. ChituDiffusion adaptations include stock-Diffusers imports, explicit
local component loading, request lifecycle separation, and EPE integration.
The complete license text is included in `LICENSE-APACHE-2.0`.
