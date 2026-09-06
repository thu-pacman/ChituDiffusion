from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.distributed as dist

from ...parallel.cp import ActiveLaneTopology, EpeParallelContext
from .conditioning import MiniMaxH3Conditioning
from .fl2va_packed_sequence import FL2VAPackedSequence
from .packed_sequence import MiniMaxH3PackedSequence, build_packed_sequence
from .transformer import (
    MiniMaxH3DiTModel,
    MiniMaxH3PackedForwardInputs,
)


def minimax_h3_time_shift_sigmas(
    num_inference_steps: int,
    *,
    shift: float,
    device: torch.device | str,
) -> torch.Tensor:
    """Return the shifted H3 sigma points, from one through zero."""

    if num_inference_steps < 2:
        raise ValueError("num_inference_steps must be at least 2")
    if shift <= 0:
        raise ValueError("flow shift must be positive")
    base = torch.linspace(
        1.0, 0.0, num_inference_steps, device=device, dtype=torch.float32
    )
    return shift * base / (1.0 + (shift - 1.0) * base)


flow_match_sigmas = minimax_h3_time_shift_sigmas


def _video_to_rows(video: torch.Tensor, patch: tuple[int, int, int]) -> torch.Tensor:
    pt, ph, pw = patch
    channels, frames, height, width = video.shape
    if frames % pt or height % ph or width % pw:
        raise ValueError("video latent dimensions must be divisible by patch size")
    return (
        video.view(
            channels,
            frames // pt,
            pt,
            height // ph,
            ph,
            width // pw,
            pw,
        )
        .permute(1, 3, 5, 0, 2, 4, 6)
        .reshape(-1, channels * pt * ph * pw)
        .contiguous()
    )


def _rows_to_video(
    rows: torch.Tensor,
    shape: tuple[int, int, int, int],
    patch: tuple[int, int, int],
) -> torch.Tensor:
    channels, frames, height, width = shape
    pt, ph, pw = patch
    return (
        rows.view(
            frames // pt,
            height // ph,
            width // pw,
            channels,
            pt,
            ph,
            pw,
        )
        .permute(3, 0, 4, 1, 5, 2, 6)
        .reshape(shape)
        .contiguous()
    )


def _normalize_prompt(prompt: torch.Tensor, text_dim: int) -> torch.Tensor:
    if prompt.ndim == 3 and prompt.shape[0] == 1:
        prompt = prompt[0]
    if prompt.ndim != 2 or prompt.shape[-1] != text_dim:
        raise ValueError(
            f"prompt_embeds must have shape [text, {text_dim}] or [1, text, "
            f"{text_dim}], got {tuple(prompt.shape)}"
        )
    return prompt.contiguous()


def _normalize_video(video: torch.Tensor, channels: int) -> torch.Tensor:
    if video.ndim == 5 and video.shape[0] == 1:
        video = video[0]
    if video.ndim != 4:
        raise ValueError("video_latents must have shape [C,T,H,W] or [1,C,T,H,W]")
    if video.shape[0] != channels and video.shape[1] == channels:
        video = video.permute(1, 0, 2, 3)
    if video.shape[0] != channels:
        raise ValueError(f"video_latents channel dimension must be {channels}")
    return video.contiguous()


def _normalize_audio(audio: torch.Tensor, latent_dim: int) -> torch.Tensor:
    if audio.ndim == 4 and audio.shape[0] == 1:
        audio = audio[0]
    if audio.ndim != 3:
        raise ValueError("audio_latents must have shape [channels,time,dim]")
    if audio.shape[-1] != latent_dim and audio.shape[1] == latent_dim:
        audio = audio.transpose(1, 2)
    if audio.shape[-1] != latent_dim:
        raise ValueError(f"audio_latents final dimension must be {latent_dim}")
    return audio.contiguous()


@dataclass(slots=True)
class MiniMaxH3DenoiseState:
    video_latents: torch.Tensor
    audio_latents: torch.Tensor
    refined_prompt_embeds: torch.Tensor
    packed: MiniMaxH3PackedSequence | FL2VAPackedSequence
    video_sigmas: torch.Tensor
    audio_sigmas: torch.Tensor
    width: int
    height: int
    image_tokens: int
    actual_batch_size: int = 1
    step_index: int = 0
    condition_video_rows: torch.Tensor | None = None
    output_type: str = "latent"
    fps: int = 24
    frame_count: int | None = None

    @property
    def latents(self) -> torch.Tensor:
        """Compatibility view for generic lifecycle code; transfers use both tensors."""

        return self.video_latents

    @property
    def timesteps(self) -> torch.Tensor:
        return 1.0 - self.video_sigmas[:-1]

    @property
    def sigmas(self) -> torch.Tensor:
        """Compatibility alias for the video schedule."""

        return self.video_sigmas

    @property
    def complete(self) -> bool:
        return self.step_index >= self.video_sigmas.numel() - 1


class MiniMaxH3LatentPipeline:
    """Resumable MiniMax-H3 DiT loop with no text/audio/video decoder dependency."""

    def __init__(
        self,
        transformer: MiniMaxH3DiTModel,
        parallel_context: EpeParallelContext,
        *,
        flow_shift: float = 12.0,
        audio_flow_shift: float = 3.0,
        attention_mode: str = "ulysses",
    ) -> None:
        self.transformer = transformer
        self._parallel_context = parallel_context
        self.flow_shift = float(flow_shift)
        self.audio_flow_shift = float(audio_flow_shift)
        if attention_mode not in {"ulysses", "agkv"}:
            raise ValueError("attention_mode must be 'ulysses' or 'agkv'")
        self.attention_mode = attention_mode

    @property
    def parallel_context(self) -> EpeParallelContext:
        return self._parallel_context

    @property
    def device(self) -> torch.device:
        return next(self.transformer.parameters()).device

    @torch.inference_mode()
    def prepare_request(
        self,
        conditioning: MiniMaxH3Conditioning,
        *,
        num_inference_steps: int,
    ) -> MiniMaxH3DenoiseState:
        config = self.transformer.config
        values = conditioning.to(self.device)
        prompt = _normalize_prompt(values.prompt_embeds, config.text_dim)
        video = _normalize_video(values.video_latents, config.latents_dim)
        audio = _normalize_audio(values.audio_latents, config.audio_latents_dim)
        _, latent_t, latent_h, latent_w = video.shape
        audio_channels, audio_t, _ = audio.shape
        packed = build_packed_sequence(
            text_length=prompt.shape[0],
            latent_t=latent_t,
            latent_h=latent_h,
            latent_w=latent_w,
            audio_t=audio_t,
            audio_channels=audio_channels,
        )
        refined = self.transformer.refine_prompt_embeds(prompt)
        video_sigmas = minimax_h3_time_shift_sigmas(
            num_inference_steps, shift=self.flow_shift, device=self.device
        )
        audio_sigmas = minimax_h3_time_shift_sigmas(
            num_inference_steps, shift=self.audio_flow_shift, device=self.device
        )
        image_tokens = latent_t * (latent_h // 2) * (latent_w // 2)
        return MiniMaxH3DenoiseState(
            video_latents=video.float(),
            audio_latents=audio.float(),
            refined_prompt_embeds=refined,
            packed=packed,
            video_sigmas=video_sigmas,
            audio_sigmas=audio_sigmas,
            width=latent_w,
            height=latent_h,
            image_tokens=image_tokens,
        )

    @torch.inference_mode()
    def prepare_native_request(
        self,
        *,
        prompt_embeds: torch.Tensor,
        video_noise: torch.Tensor,
        audio_rows: torch.Tensor,
        packed: FL2VAPackedSequence,
        num_inference_steps: int,
        condition_video_rows: torch.Tensor | None = None,
        output_type: str = "mp4",
        fps: int = 24,
        frame_count: int | None = None,
    ) -> MiniMaxH3DenoiseState:
        """Prepare native FL2VA state from encoder and geometry contracts."""

        config = self.transformer.config
        prompt = _normalize_prompt(prompt_embeds.to(self.device), config.text_dim)
        video = _normalize_video(video_noise.to(self.device), config.latents_dim)
        if audio_rows.ndim != 2 or audio_rows.shape[-1] != config.audio_latents_dim:
            raise ValueError("native audio noise must be [rows, audio_latents_dim]")
        audio_row_count = packed.audio_slice.stop - packed.audio_slice.start
        if audio_rows.shape[0] != audio_row_count:
            raise ValueError("native audio noise row count does not match packed layout")
        audio_channels = 2
        if audio_row_count % audio_channels:
            raise ValueError("native audio row count must contain stereo channels")
        audio = (
            audio_rows.to(self.device)
            .reshape(audio_channels, audio_row_count // audio_channels, -1)
            .contiguous()
        )
        condition_count = packed.condition_slice.stop - packed.condition_slice.start
        if condition_count:
            if condition_video_rows is None or condition_video_rows.shape[0] != condition_count:
                raise ValueError("keyframe packed layout requires encoded condition rows")
        elif condition_video_rows is not None and condition_video_rows.numel():
            raise ValueError("prompt-only packed layout must not contain condition rows")
        refined = self.transformer.refine_prompt_embeds(prompt)
        return MiniMaxH3DenoiseState(
            video_latents=video.float(),
            audio_latents=audio.float(),
            refined_prompt_embeds=refined,
            packed=packed,
            video_sigmas=minimax_h3_time_shift_sigmas(
                num_inference_steps, shift=self.flow_shift, device=self.device
            ),
            audio_sigmas=minimax_h3_time_shift_sigmas(
                num_inference_steps, shift=self.audio_flow_shift, device=self.device
            ),
            width=video.shape[-1],
            height=video.shape[-2],
            image_tokens=packed.sequence_length,
            condition_video_rows=(
                None
                if condition_video_rows is None
                else condition_video_rows.to(self.device).float().contiguous()
            ),
            output_type=output_type,
            fps=int(fps),
            frame_count=frame_count,
        )

    def _full_prediction(
        self,
        state: MiniMaxH3DenoiseState,
        *,
        topology: ActiveLaneTopology,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        packed = state.packed
        shard_value = packed.shard(topology.rank_in_lane, topology.width)
        shard = (
            shard_value
            if isinstance(shard_value, slice)
            else slice(shard_value.start, shard_value.stop)
        )
        generated_video_rows = _video_to_rows(
            state.video_latents, self.transformer.config.patch_size
        )
        video_rows = (
            generated_video_rows
            if state.condition_video_rows is None
            else torch.cat(
                (
                    state.condition_video_rows.to(
                        device=self.device, dtype=generated_video_rows.dtype
                    ),
                    generated_video_rows,
                )
            )
        )
        audio_rows = state.audio_latents.reshape(
            -1, self.transformer.config.audio_latents_dim
        )
        row_timesteps = (1.0 - state.video_sigmas[state.step_index]).expand(
            packed.sequence_length
        ).clone()
        row_timesteps[packed.audio_pos.to(self.device)] = (
            1.0 - state.audio_sigmas[state.step_index]
        )
        if isinstance(packed, FL2VAPackedSequence):
            row_timesteps[packed.condition_slice] = 0.0
        unique_timesteps, inverse_indices = torch.unique(
            row_timesteps, sorted=True, return_inverse=True
        )
        video, audio = self.transformer.forward_packed(
            MiniMaxH3PackedForwardInputs(
            video_rows=video_rows,
            audio_rows=audio_rows,
            refined_prompt_embeds=state.refined_prompt_embeds,
            img_pos=packed.img_pos.to(self.device),
            audio_pos=packed.audio_pos.to(self.device),
            text_pos=packed.text_pos.to(self.device),
            global_sequence_length=packed.sequence_length,
            row_start=shard.start,
            row_stop=shard.stop,
            unique_timesteps=unique_timesteps,
            inverse_indices=inverse_indices,
            token_tags=packed.token_tags.to(self.device),
            position_ids=packed.position_ids.to(self.device),
            cu_seqlens=packed.cu_seqlens.to(self.device),
            max_seqlen=packed.max_seqlen,
            ),
            ulysses_topology=(
                replace(
                    self.parallel_context.active_usp,
                    attention_mode=self.attention_mode,
                )
                if topology.width > 1
                else None
            ),
            gather_tp_output=True,
        )
        if topology.width > 1:
            video_parts = [torch.empty_like(video) for _ in range(topology.width)]
            audio_parts = [torch.empty_like(audio) for _ in range(topology.width)]
            dist.all_gather(
                video_parts, video.contiguous(), group=topology.process_group
            )
            dist.all_gather(
                audio_parts, audio.contiguous(), group=topology.process_group
            )
            video = torch.cat(video_parts, dim=0)
            audio = torch.cat(audio_parts, dim=0)
        return video, audio

    @torch.inference_mode()
    def denoise_step(
        self,
        state: MiniMaxH3DenoiseState,
        *,
        lane_ranks: tuple[int, ...],
    ) -> MiniMaxH3DenoiseState:
        if state.complete:
            raise RuntimeError("denoise state is already complete")
        plane_lane = self.parallel_context.plane_lane(lane_ranks)
        with self.parallel_context.activate(plane_lane) as topology:
            video_prediction, audio_prediction = self._full_prediction(
                state, topology=topology
            )
            packed = state.packed
            video_rows = video_prediction.index_select(
                0, packed.target_img_pos.to(self.device)
            )
            audio_rows = audio_prediction.index_select(
                0, packed.audio_pos.to(self.device)
            )
            video_velocity = _rows_to_video(
                video_rows,
                tuple(state.video_latents.shape),
                self.transformer.config.patch_size,
            )
            audio_velocity = audio_rows.reshape_as(state.audio_latents)
            video_delta = (
                state.video_sigmas[state.step_index]
                - state.video_sigmas[state.step_index + 1]
            )
            audio_delta = (
                state.audio_sigmas[state.step_index]
                - state.audio_sigmas[state.step_index + 1]
            )
            state.video_latents = (
                state.video_latents + video_delta * video_velocity.float()
            ).contiguous()
            state.audio_latents = (
                state.audio_latents + audio_delta * audio_velocity.float()
            ).contiguous()
            state.step_index += 1
        return state

    def synchronize_state(
        self, state: MiniMaxH3DenoiseState, *, step_index: int
    ) -> None:
        index = int(step_index)
        if not 0 <= index <= state.video_sigmas.numel() - 1:
            raise ValueError("step_index is outside the sigma schedule")
        state.step_index = index

    @torch.inference_mode()
    def finalize_latent(
        self,
        state: MiniMaxH3DenoiseState,
        *,
        topology: ActiveLaneTopology | None = None,
        timings: dict[str, Any] | None = None,
    ) -> torch.Tensor | None:
        if not state.complete:
            raise RuntimeError("cannot finalize an incomplete MiniMax-H3 state")
        active = topology or self.parallel_context.active
        if not active.is_leader:
            return None
        if timings is not None:
            timings.setdefault("vae_start_unix_ns", 0)
            timings.setdefault("vae_end_unix_ns", 0)
            timings.setdefault("vae_decode_ms", 0.0)
        return pack_latent_tensors(state.video_latents, state.audio_latents)

    def maybe_free_model_hooks(self) -> None:
        return None

    def close(self) -> None:
        self.parallel_context.close()


_PACK_HEADER_VALUES = 10


def pack_latent_tensors(video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
    """Pack two shaped float32 tensors into one host-transfer tensor."""

    if video.ndim > 4 or audio.ndim > 3:
        raise ValueError("latent rank exceeds the stable package header")
    header = torch.zeros(_PACK_HEADER_VALUES, device=video.device, dtype=torch.float32)
    header[0] = video.ndim
    header[1 : 1 + video.ndim] = torch.tensor(
        video.shape, device=video.device, dtype=torch.float32
    )
    header[5] = audio.ndim
    header[6 : 6 + audio.ndim] = torch.tensor(
        audio.shape, device=video.device, dtype=torch.float32
    )
    header[9] = video.numel()
    return torch.cat((header, video.float().flatten(), audio.float().flatten()))


def unpack_latent_tensors(packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if packed.ndim != 1 or packed.numel() < _PACK_HEADER_VALUES:
        raise ValueError("invalid packed MiniMax-H3 latent tensor")
    video_ndim = int(packed[0].item())
    audio_ndim = int(packed[5].item())
    video_shape = tuple(int(value) for value in packed[1 : 1 + video_ndim].tolist())
    audio_shape = tuple(int(value) for value in packed[6 : 6 + audio_ndim].tolist())
    video_count = int(packed[9].item())
    video_start = _PACK_HEADER_VALUES
    audio_start = video_start + video_count
    video = packed[video_start:audio_start].reshape(video_shape).contiguous()
    audio = packed[audio_start:].reshape(audio_shape).contiguous()
    return video, audio
