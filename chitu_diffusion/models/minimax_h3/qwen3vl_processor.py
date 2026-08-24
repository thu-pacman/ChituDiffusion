from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch


VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"


@dataclass(frozen=True, slots=True)
class MiniMaxH3Qwen3VLInputs:
    input_ids: torch.Tensor
    token_type_ids: torch.Tensor
    position_ids: torch.Tensor
    pixel_values: torch.Tensor | None = None
    image_grid_thw: torch.Tensor | None = None

    @property
    def token_tags(self) -> torch.Tensor:
        return self.token_type_ids


def _encode_text(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    return list(encoded["input_ids"])


def build_h3_presentation(
    tokenizer: Any,
    prompt: str,
    *,
    image_token_counts: Sequence[int] = (),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the official H3 presentation without applying a chat template."""

    if not prompt:
        raise ValueError("prompt must be non-empty")
    counts = [int(value) for value in image_token_counts]
    if len(counts) > 2:
        raise ValueError("FL2VA accepts at most first and last images")
    if not counts:
        ids = _encode_text(tokenizer, prompt)
        return torch.tensor(ids, dtype=torch.long), torch.ones(
            len(ids), dtype=torch.long
        )

    vision_start = int(tokenizer.convert_tokens_to_ids(VISION_START))
    vision_end = int(tokenizer.convert_tokens_to_ids(VISION_END))
    image_pad = int(tokenizer.convert_tokens_to_ids(IMAGE_PAD))
    ids: list[int] = []
    tags: list[int] = []
    for count in counts:
        if count <= 0:
            raise ValueError("image token counts must be positive")
        label = _encode_text(tokenizer, ": ")
        block = [vision_start, *([image_pad] * count), vision_end]
        text = _encode_text(tokenizer, prompt)
        ids.extend(label)
        tags.extend([1] * len(label))
        ids.extend(block)
        tags.extend([0] * len(block))
        ids.extend(text)
        tags.extend([1] * len(text))
    return torch.tensor(ids, dtype=torch.long), torch.tensor(tags, dtype=torch.long)


def build_qwen3vl_position_ids(
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor | None,
    *,
    image_token_id: int,
    spatial_merge_size: int,
) -> torch.Tensor:
    """Build Qwen3-VL temporal/height/width positions for one presentation."""

    if input_ids.ndim != 1:
        raise ValueError("H3 position IDs require one unbatched presentation")
    if image_grid_thw is None:
        positions = torch.arange(input_ids.numel(), dtype=torch.long)
        return positions.view(1, -1).expand(3, -1).clone()
    grids = image_grid_thw.to(device="cpu", dtype=torch.long)
    if grids.ndim != 2 or grids.shape[1] != 3:
        raise ValueError("image_grid_thw must have shape [images, 3]")
    token_list = input_ids.to(device="cpu", dtype=torch.long).tolist()
    pieces: list[torch.Tensor] = []
    start = 0
    next_position = 0
    for grid in grids:
        try:
            image_start = token_list.index(int(image_token_id), start)
        except ValueError as exc:
            raise ValueError(
                "presentation has fewer image placeholders than image grids"
            ) from exc
        temporal, height, width = (int(value) for value in grid)
        if height % spatial_merge_size or width % spatial_merge_size:
            raise ValueError("vision grid must be divisible by spatial merge size")
        grid_h = height // spatial_merge_size
        grid_w = width // spatial_merge_size
        visual_tokens = temporal * grid_h * grid_w
        stop = image_start + visual_tokens
        if token_list[image_start:stop] != [int(image_token_id)] * visual_tokens:
            raise ValueError(
                "image placeholder count does not match the processed vision grid"
            )
        text_length = image_start - start
        if text_length:
            text = torch.arange(text_length).view(1, -1).expand(3, -1)
            pieces.append(text + next_position)
            next_position += text_length
        t = (
            torch.arange(temporal)
            .view(-1, 1, 1)
            .expand(-1, grid_h, grid_w)
            .flatten()
        )
        h = (
            torch.arange(grid_h)
            .view(1, -1, 1)
            .expand(temporal, -1, grid_w)
            .flatten()
        )
        w = (
            torch.arange(grid_w)
            .view(1, 1, -1)
            .expand(temporal, grid_h, -1)
            .flatten()
        )
        visual = torch.stack((t, h, w)) + next_position
        pieces.append(visual)
        next_position = int(visual.max()) + 1
        start = stop
    if int((input_ids == image_token_id).sum()) != sum(
        int(grid.prod()) // spatial_merge_size**2 for grid in grids
    ):
        raise ValueError("unused or missing image placeholder tokens")
    if start < input_ids.numel():
        tail = torch.arange(input_ids.numel() - start).view(1, -1).expand(3, -1)
        pieces.append(tail + next_position)
    position_ids = torch.cat(pieces, dim=1)
    if position_ids.shape != (3, input_ids.numel()):
        raise ValueError("3D position IDs do not align with presentation tokens")
    return position_ids


class MiniMaxH3Qwen3VLProcessor:
    """Exact H3 presentation plus Qwen3-VL first/last-frame preprocessing."""

    vision_supported = True

    def __init__(
        self,
        tokenizer: Any,
        image_processor: Any | None = None,
        *,
        spatial_merge_size: int = 2,
    ) -> None:
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.spatial_merge_size = int(spatial_merge_size)

    @classmethod
    def from_pretrained(
        cls, checkpoint_dir: str | Path
    ) -> MiniMaxH3Qwen3VLProcessor:
        # Keep Transformers optional for prompt-only import and test paths.
        from transformers import AutoImageProcessor, AutoProcessor

        processor = AutoProcessor.from_pretrained(
            checkpoint_dir,
            local_files_only=True,
            trust_remote_code=False,
        )
        tokenizer = getattr(processor, "tokenizer", processor)
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is None:
            image_processor = AutoImageProcessor.from_pretrained(
                checkpoint_dir,
                local_files_only=True,
                trust_remote_code=False,
            )
        merge_size = int(getattr(image_processor, "merge_size", 2))
        return cls(
            tokenizer,
            image_processor,
            spatial_merge_size=merge_size,
        )

    def __call__(
        self,
        prompt: str,
        *,
        first_image: Any | None = None,
        last_image: Any | None = None,
    ) -> MiniMaxH3Qwen3VLInputs:
        images = [
            image
            for image in (first_image, last_image)
            if image is not None
        ]
        pixel_values = None
        image_grid_thw = None
        counts: list[int] = []
        if images:
            if self.image_processor is None:
                raise RuntimeError("image_processor is required for first/last frames")
            processed = self.image_processor(images=images, return_tensors="pt")
            pixel_values = torch.as_tensor(processed["pixel_values"])
            image_grid_thw = torch.as_tensor(
                processed["image_grid_thw"], dtype=torch.long
            )
            if image_grid_thw.shape != (len(images), 3):
                raise ValueError("image processor returned an invalid image_grid_thw")
            counts = [
                int(grid.prod()) // self.spatial_merge_size**2
                for grid in image_grid_thw
            ]
        input_ids, token_type_ids = build_h3_presentation(
            self.tokenizer, prompt, image_token_counts=counts
        )
        image_token_id = int(self.tokenizer.convert_tokens_to_ids(IMAGE_PAD))
        position_ids = build_qwen3vl_position_ids(
            input_ids,
            image_grid_thw,
            image_token_id=image_token_id,
            spatial_merge_size=self.spatial_merge_size,
        )
        return MiniMaxH3Qwen3VLInputs(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids.unsqueeze(1),
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

    def build_preprocessed_image_presentation(
        self, prompt: str, image_token_counts: Sequence[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expose the exact FL2VA token contract for vision-adapter bring-up."""

        return build_h3_presentation(
            self.tokenizer, prompt, image_token_counts=image_token_counts
        )


__all__ = [
    "MiniMaxH3Qwen3VLInputs",
    "MiniMaxH3Qwen3VLProcessor",
    "build_h3_presentation",
    "build_qwen3vl_position_ids",
]
