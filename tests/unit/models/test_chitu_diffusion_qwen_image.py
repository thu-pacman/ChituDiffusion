from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from diffusers.models.attention_processor import Attention

from chitu_diffusion import QwenImageRequest
from chitu_diffusion.epe.api import DiffusersEPEPipeline
from chitu_diffusion.epe.scheduling.planner import EpeSchedulingModule
from chitu_diffusion.models.qwen_image.api import QwenImagePipeline
from chitu_diffusion.models.qwen_image.attention import QwenImageCpAttnProcessor
from chitu_diffusion.models.qwen_image.executor import QwenImageDecoderExecutor
from chitu_diffusion.models.qwen_image.pipeline import (
    EpeQwenImagePipeline,
    combine_qwen_cfg_predictions,
)
from chitu_diffusion.models.qwen_image.transformer import (
    EpeQwenImageTransformer2DModel,
)
from chitu_diffusion.models.zimage.attention import ZImageCpAttnProcessor
from chitu_diffusion.parallel.vae import VaeParallelPlacement


def test_qwen_image_request_defaults_to_true_cfg() -> None:
    request = QwenImageRequest(prompt="test", width=512, height=768, num_steps=50)

    assert request.negative_prompt == " "
    assert request.true_cfg_scale == 4.0
    assert request.guidance_scale is None

    with pytest.raises(ValueError, match="multiples of 16"):
        QwenImageRequest(prompt="test", width=510, height=512)


def test_qwen_image_request_builds_diffusion_inputs() -> None:
    request = QwenImageRequest(
        prompt="test",
        negative_prompt="bad",
        width=512,
        height=512,
        num_steps=4,
        true_cfg_scale=3.0,
        seed=7,
    )

    diffusion = request.to_diffusion_request(device=torch.device("cpu"))

    assert diffusion.inputs["negative_prompt"] == "bad"
    assert diffusion.inputs["true_cfg_scale"] == 3.0
    assert diffusion.inputs["guidance_scale"] is None
    assert diffusion.num_inference_steps == 4


def test_qwen_cfg_combination_matches_native_formula() -> None:
    positive = torch.tensor([[[3.0, 4.0]]])
    negative = torch.tensor([[[1.0, 2.0]]])

    output = combine_qwen_cfg_predictions(positive, negative, true_cfg_scale=4.0)
    combined = negative + 4.0 * (positive - negative)
    expected = combined * (
        torch.norm(positive, dim=-1, keepdim=True)
        / torch.norm(combined, dim=-1, keepdim=True)
    )

    assert torch.equal(output, expected)


def test_qwen_image_uses_shared_api_layer() -> None:
    assert issubclass(QwenImagePipeline, DiffusersEPEPipeline)


def test_qwen_image_finalize_skips_postprocess_on_nonleader() -> None:
    pipeline = SimpleNamespace(
        decode_request=lambda state: None,
        image_processor=SimpleNamespace(
            postprocess=lambda *args, **kwargs: pytest.fail(
                "nonleader output must not be postprocessed"
            )
        ),
        maybe_free_model_hooks=lambda: None,
    )
    state = SimpleNamespace(latents=torch.empty(1))

    output = EpeQwenImagePipeline.finalize_request(pipeline, state)

    assert output.images is None


def test_qwen_image_executor_profiles_cfg_and_packed_state() -> None:
    class FakeParallel:
        rank = 0
        local_rank = 0
        world_size = 1
        allowed_widths = (1,)

    class FakePipeline:
        parallel_context = FakeParallel()
        transformer = torch.nn.Linear(4, 4, bias=False)

    executor = QwenImageDecoderExecutor(
        FakePipeline(),
        EpeSchedulingModule(FakeParallel()),
        default_width=512,
        default_height=512,
        default_num_steps=50,
        vae_placement=VaeParallelPlacement(halo=4, sharded=False),
    )
    request = executor.normalize_request({"prompt": "test", "guidance_scale": 2.5})
    profile = executor.request_profile(request, completed_steps=3)

    assert profile.image_tokens == 1024
    assert profile.attributes == {
        "batch_size": 1,
        "conditions": 2,
        "state_bytes": 1024 * 64 * 4,
    }
    assert (executor.parallel_vae, executor.vae_parallel_halo) == (False, 4)
    assert request.true_cfg_scale == 2.5


def test_joint_attention_packed_projection_cache_is_per_layer() -> None:
    qwen_processor = QwenImageCpAttnProcessor(SimpleNamespace(), sharded_text=True)
    qwen_attentions = [
        Attention(
            query_dim=16,
            added_kv_proj_dim=16,
            dim_head=4,
            heads=4,
            out_dim=16,
            context_pre_only=False,
            bias=True,
            qk_norm="rms_norm",
        )
        for _ in range(2)
    ]
    first = qwen_processor._packed_qkv_for(qwen_attentions[0])
    second = qwen_processor._packed_qkv_for(qwen_attentions[1])

    assert qwen_processor._packed_qkv_for(qwen_attentions[0])[0] is first[0]
    assert second[0] is not first[0]

    z_processor = ZImageCpAttnProcessor(SimpleNamespace(), sharded_text=True)
    z_attention = Attention(
        query_dim=16,
        dim_head=4,
        heads=4,
        qk_norm="rms_norm",
        bias=False,
    )
    packed_weight, _ = z_processor._packed_qkv_for(z_attention)
    with torch.no_grad():
        z_attention.to_q.weight.add_(1)
    refreshed_weight, _ = z_processor._packed_qkv_for(z_attention)

    assert refreshed_weight is not packed_weight


def test_joint_attention_packed_projection_cache_accepts_inference_tensors() -> None:
    qwen_processor = QwenImageCpAttnProcessor(SimpleNamespace(), sharded_text=True)
    z_processor = ZImageCpAttnProcessor(SimpleNamespace(), sharded_text=True)
    with torch.inference_mode():
        qwen_attention = Attention(
            query_dim=16,
            added_kv_proj_dim=16,
            dim_head=4,
            heads=4,
            out_dim=16,
            context_pre_only=False,
            bias=True,
            qk_norm="rms_norm",
        )
        z_attention = Attention(
            query_dim=16,
            dim_head=4,
            heads=4,
            qk_norm="rms_norm",
            bias=False,
        )
        qwen_first = qwen_processor._packed_qkv_for(qwen_attention)
        qwen_second = qwen_processor._packed_qkv_for(qwen_attention)
        z_first = z_processor._packed_qkv_for(z_attention)
        z_second = z_processor._packed_qkv_for(z_attention)

    assert qwen_second[0] is qwen_first[0]
    assert z_second[0] is z_first[0]


def test_qwen_fast_cp_configures_one_planned_processor_per_layer() -> None:
    transport = SimpleNamespace(name="fast_agkv")
    parallel = SimpleNamespace(
        active=SimpleNamespace(width=2),
        active_agkv_transport=transport,
        active_ulysses=SimpleNamespace(transport=transport),
    )
    model = EpeQwenImageTransformer2DModel(
        patch_size=2,
        in_channels=8,
        out_channels=4,
        num_layers=2,
        attention_head_dim=8,
        num_attention_heads=4,
        joint_attention_dim=16,
        axes_dims_rope=(2, 2, 4),
    )

    model.configure_epe(parallel, attention_mode="agkv")

    processors = model._epe_attn_processors
    assert len(processors) == len({id(processor) for processor in processors}) == 2
    assert all(processor.attention.planned for processor in processors)
    assert all(processor.sharded_text for processor in processors)
