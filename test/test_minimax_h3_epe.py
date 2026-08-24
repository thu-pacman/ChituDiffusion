from __future__ import annotations

import torch
from types import SimpleNamespace

from chitu_diffusion.epac.model_scheduling import EpeSchedulingModule
from chitu_diffusion.models.minimax_h3 import (
    MiniMaxH3DiTConfig,
    MiniMaxH3DiTModel,
    MiniMaxH3LatentExecutor,
    MiniMaxH3LatentPipeline,
    make_synthetic_conditioning,
    minimax_h3_time_shift_sigmas,
    pack_latent_tensors,
    unpack_latent_tensors,
)
from chitu_diffusion.parallel.groups import EpeParallelContext
from chitu_diffusion.parallel.tensor_parallel import TensorParallelTopology
from chitu_diffusion.serve.protocol import ImageGenerateRequest
from chitu_diffusion.epac.image_decoder import TerminalArtifact


def _context() -> EpeParallelContext:
    topology = TensorParallelTopology(
        ranks=(0,),
        rank=0,
        rank_in_group=0,
        degree=1,
        process_group=None,
    )
    return EpeParallelContext(
        rank=0,
        world_size=1,
        local_rank=0,
        allowed_widths=(1,),
        groups={(0,): None},
        usp_topologies={},
        cfg_topologies={},
        tensor_parallel_topology=topology,
        initial_lane_ranks=(0,),
        owned_groups=(),
        owns_world=False,
    )


def _tiny_config() -> MiniMaxH3DiTConfig:
    return MiniMaxH3DiTConfig(
        hidden_size=16,
        num_layers=1,
        token_refiner_num_layers=1,
        num_attention_heads=2,
        attention_head_dim=8,
        ffn_hidden_size=32,
        latents_dim=2,
        audio_latents_dim=4,
        patch_size=(1, 2, 2),
        text_dim=12,
        timestep_input_dim=8,
        time_embed_hidden_size=16,
        time_embed_dim=8,
        adaln_out_features=18 * 16,
        final_adaln_out_features=2 * 16,
        rope_inv_freq_len=1,
    )


def test_h3_time_shift_schedule_matches_sglang_contract() -> None:
    sigmas = minimax_h3_time_shift_sigmas(4, shift=12.0, device="cpu")
    base = torch.linspace(1.0, 0.0, 4, dtype=torch.float32)
    expected = 12.0 * base / (1.0 + 11.0 * base)

    torch.testing.assert_close(sigmas, expected)
    assert sigmas.numel() - 1 == 3


def test_h3_executor_merges_http_model_inputs_and_aliases() -> None:
    context = _context()
    try:
        model = MiniMaxH3DiTModel(_tiny_config(), attention_backend="sdpa")
        pipeline = MiniMaxH3LatentPipeline(model, context)
        executor = MiniMaxH3LatentExecutor(
            pipeline,
            EpeSchedulingModule(context),
        )
        request = executor.normalize_request(
            ImageGenerateRequest(
                prompt="offline",
                seed=7,
                model_inputs={
                    "synthetic": True,
                    "latent_h": 4,
                    "latent_w": 6,
                    "latent_t": 1,
                    "audio_t": 1,
                    "audio_channels": 2,
                    "text_length": 2,
                    "num_steps": 3,
                },
            )
        )

        assert request.synthetic
        assert (request.height, request.width, request.seed) == (4, 6, 7)
    finally:
        context.close()


def test_h3_tiny_latent_pipeline_runs_to_completion() -> None:
    context = _context()
    try:
        config = _tiny_config()
        model = MiniMaxH3DiTModel(config, attention_backend="sdpa")
        pipeline = MiniMaxH3LatentPipeline(model, context)
        conditioning = make_synthetic_conditioning(
            text_length=2,
            text_dim=config.text_dim,
            latent_t=1,
            latent_h=4,
            latent_w=4,
            latent_channels=config.latents_dim,
            audio_t=1,
            audio_channels=2,
            audio_latent_dim=config.audio_latents_dim,
            seed=11,
        )
        state = pipeline.prepare_request(conditioning, num_inference_steps=3)

        while not state.complete:
            pipeline.denoise_step(state, lane_ranks=(0,))

        assert state.step_index == 2
        assert torch.isfinite(state.video_latents).all()
        assert torch.isfinite(state.audio_latents).all()
        packed = pack_latent_tensors(state.video_latents, state.audio_latents)
        video, audio = unpack_latent_tensors(packed)
        torch.testing.assert_close(video, state.video_latents)
        torch.testing.assert_close(audio, state.audio_latents)
    finally:
        context.close()


def test_h3_migration_allocation_skips_conditioning_and_exports_static_state(
    monkeypatch,
) -> None:
    context = _context()
    try:
        model = MiniMaxH3DiTModel(_tiny_config(), attention_backend="sdpa")
        pipeline = MiniMaxH3LatentPipeline(model, context)
        executor = MiniMaxH3LatentExecutor(
            pipeline,
            EpeSchedulingModule(context),
        )
        monkeypatch.setattr(
            pipeline,
            "prepare_request",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("migration allocation must not initialize")
            ),
        )
        state = executor.allocate_request(
            {
                "synthetic": True,
                "width": 4,
                "height": 4,
                "latent_t": 1,
                "audio_t": 1,
                "audio_channels": 2,
                "text_length": 2,
                "num_steps": 3,
            }
        )
        bundle = executor.export_state(state)
        assert tuple(bundle.tensors) == (
            "video_latents",
            "audio_latents",
            "refined_prompt_embeds",
            "video_sigmas",
            "audio_sigmas",
        )
    finally:
        context.close()


def test_h3_native_prompt_only_prepare_and_structured_finalize() -> None:
    class Processor:
        def __call__(self, prompt, **kwargs):
            assert prompt == "a test"
            assert not any(kwargs.values())
            return SimpleNamespace(input_ids=torch.tensor([1, 2, 3]))

    class Encoder(torch.nn.Module):
        def forward(self, input_ids, **kwargs):
            assert not kwargs
            return torch.zeros(input_ids.numel(), 12, dtype=torch.bfloat16)

    class VideoVAE:
        def decode(self, latents):
            assert latents.ndim == 5
            return torch.zeros(1, 3, 5, 32, 32)

    class AudioVAE:
        sample_rate = 240

        def decode(self, latents):
            assert latents.ndim == 3
            return torch.zeros(1, 2, 100)

    context = _context()
    try:
        model = MiniMaxH3DiTModel(_tiny_config(), attention_backend="sdpa")
        pipeline = MiniMaxH3LatentPipeline(model, context)
        executor = MiniMaxH3LatentExecutor(
            pipeline,
            EpeSchedulingModule(context),
            text_processor=Processor(),
            text_encoder=Encoder(),
            video_vae=VideoVAE(),
            audio_vae=AudioVAE(),
            native_enabled=True,
        )
        state = executor.prepare_request(
            {
                "prompt": "a test",
                "width": 32,
                "height": 32,
                "duration_s": 0.1,
                "fps": 24,
                "num_steps": 2,
                "output_type": "mp4",
            }
        )
        assert state.refined_prompt_embeds.shape == (3, 16)
        assert state.frame_count == 5
        state.step_index = state.video_sigmas.numel() - 1
        artifact = executor.finalize_gpu(state, lane_ranks=(0,), timings={})
        assert isinstance(artifact, TerminalArtifact)
        assert set(artifact.tensors) == {"video", "audio"}
        assert artifact.tensors["audio"].shape == (1, 2, 50)
        assert artifact.metadata["output_type"] == "mp4"
        assert artifact.metadata["sample_rate"] == 240
    finally:
        context.close()


def test_h3_native_request_profile_uses_real_presentation_length() -> None:
    class Processor:
        def __call__(self, prompt, **kwargs):
            assert prompt == "variable prompt"
            assert not any(kwargs.values())
            return SimpleNamespace(input_ids=torch.tensor([1, 2, 3, 4, 5]))

    context = _context()
    try:
        model = MiniMaxH3DiTModel(_tiny_config(), attention_backend="sdpa")
        pipeline = MiniMaxH3LatentPipeline(model, context)
        executor = MiniMaxH3LatentExecutor(
            pipeline,
            EpeSchedulingModule(context),
            text_processor=Processor(),
            native_enabled=True,
            default_text_length=32,
        )
        profile = executor.request_profile(
            {
                "request_id": "profile-real-text",
                "prompt": "variable prompt",
                "width": 32,
                "height": 32,
                "duration_s": 0.1,
                "fps": 24,
                "num_steps": 2,
            },
            completed_steps=0,
        )

        features = profile.attributes["cost_features"]
        assert features["text_tokens"] == 5
        assert profile.image_tokens == features["padded_tokens"]
        assert features["used_tokens"] + features["pad_tokens"] == profile.image_tokens
    finally:
        context.close()
