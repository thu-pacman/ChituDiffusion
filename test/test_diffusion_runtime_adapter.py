from types import SimpleNamespace

import torch

from chitu_diffusion.core.config_loader import load_config
from chitu_diffusion.runtime.backend import DiffusionBackend
from chitu_diffusion.runtime.generator import Generator
from chitu_diffusion.runtime.adapter import get_model_runtime_spec
from chitu_diffusion.runtime.task import (
    DiffusionTask,
    DiffusionTaskType,
    DiffusionUserParams,
    DiffusionUserRequest,
)


def _cfg(model_name: str):
    cfg = load_config([f"models={model_name}", "models.ckpt_dir=/tmp/chitu-model"])
    return cfg


def test_runtime_specs_resolve_checkpoint_paths():
    cases = {
        "Wan2.1-T2V-1.3B": ["diffusion_pytorch_model.safetensors"],
        "Wan2.2-T2V-A14B": ["high_noise_model", "low_noise_model"],
        "FLUX.1-dev": ["transformer"],
        "FLUX.2-klein-4B": ["transformer"],
    }

    for model_name, suffixes in cases.items():
        cfg = _cfg(model_name)
        spec = get_model_runtime_spec(cfg.models)
        adapter = spec.create_adapter()
        paths = adapter.checkpoint_paths(cfg)
        assert len(paths) == len(suffixes)
        for path, suffix in zip(paths, suffixes):
            assert path.endswith(suffix)


def test_generator_lifecycle_delegates_to_runtime_adapter(monkeypatch):
    calls = []

    class MockAdapter:
        def encode_text(self, task, generator, backend):
            calls.append("encode_text")
            return torch.ones(1, 2)

        def prepare_denoise(self, task, generator, backend):
            calls.append("prepare_denoise")
            task.buffer.latents = torch.ones(1, 2)
            task.buffer.timesteps = [torch.tensor(1.0)]
            task.buffer.sampler = object()

        def denoise_step(self, task, generator, backend, run_dit_forward):
            calls.append("denoise_step")
            return task.buffer.latents + 1

        def decode_latents(self, task, generator, backend):
            calls.append("decode_latents")
            return torch.full((1, 2), 3.0)

        def save_output(self, task, output, generator, backend):
            calls.append("save_output")

    class FakeGroup:
        group_size = 1
        rank_in_group = 0

    monkeypatch.setattr(DiffusionBackend, "model_adapter", MockAdapter())
    monkeypatch.setattr(DiffusionBackend, "do_cfg", False)
    monkeypatch.setattr("chitu_diffusion.runtime.generator.get_cfg_group", lambda: FakeGroup())
    monkeypatch.setattr("chitu_diffusion.runtime.generator.log_stage", lambda *args, **kwargs: None)
    monkeypatch.setattr("chitu_diffusion.runtime.generator.log_progress", lambda *args, **kwargs: None)
    monkeypatch.setattr("chitu_diffusion.runtime.generator.log_result", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "chitu_diffusion.runtime.generator.get_global_args",
        lambda: SimpleNamespace(output=SimpleNamespace(memory=False)),
    )
    monkeypatch.setattr("chitu_diffusion.runtime.generator.Timer.print_statistics", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "chitu_diffusion.runtime.generator.Timer.save_task_statistics_json",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "chitu_diffusion.runtime.generator.timing_metrics_dir",
        lambda run_output_dir: "/tmp",
    )
    monkeypatch.setattr(
        "chitu_diffusion.runtime.generator.memory_metrics_dir",
        lambda run_output_dir: "/tmp",
    )
    monkeypatch.setattr(
        "chitu_diffusion.runtime.generator.torch.distributed.get_rank",
        lambda: 0,
    )

    generator = Generator.__new__(Generator)
    generator.cfg_size = 1
    generator.cp_size = 1
    generator.rank = 0
    generator.enable_stage_perf = False
    generator._last_logged_stage = {}
    generator._stage_start_time = {}
    generator._dit_forward_step_elapsed_ms = {}
    generator.denoise_progress_interval = 1
    generator._clear_ditango_planner = lambda: None
    generator._clear_flexcache_strategy = lambda: None

    req = DiffusionUserRequest(
        request_id="runtime-adapter",
        params=DiffusionUserParams(
            prompt="adapter test",
            seed=1,
            num_inference_steps=1,
            sample_solver="unipc",
            size=(64, 64),
        ),
    )
    task = DiffusionTask(task_id=req.request_id, req=req)

    text = generator.text_encode_step(task)
    generator._update_task_stage_and_buffer(task, text)
    assert task.task_type == DiffusionTaskType.Denoise

    denoised = generator.denoise_step(task)
    generator._update_task_stage_and_buffer(task, denoised)
    assert task.task_type == DiffusionTaskType.VAEDecode

    decoded = generator.vae_decode_step(task)
    generator._update_task_stage_and_buffer(task, decoded)

    assert calls == [
        "encode_text",
        "prepare_denoise",
        "denoise_step",
        "decode_latents",
        "save_output",
    ]
