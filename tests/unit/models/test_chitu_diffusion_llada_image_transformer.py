from __future__ import annotations

import pytest
import torch
from diffusers.models.normalization import RMSNorm as DiffusersRMSNorm

from chitu_diffusion.models.llada_image.attention import LLaDAImageCpAttnProcessor
from chitu_diffusion.models.llada_image.diffusers_components import (
    RMSNorm as LLaDAImageRMSNorm,
)
from chitu_diffusion.models.llada_image.epe import LLaDAImageEpeModule
from chitu_diffusion.models.llada_image.kernels import rms_norm_available
from chitu_diffusion.models.llada_image.transformer import (
    EpeLLaDAImageTransformer2DModel,
)
from chitu_diffusion.parallel.cp import EpeParallelContext


def _tiny_model(*, layers: int = 1) -> EpeLLaDAImageTransformer2DModel:
    return EpeLLaDAImageTransformer2DModel(
        in_channels=2,
        dim=6,
        n_layers=layers,
        n_refiner_layers=0,
        n_heads=1,
        cap_feat_dim=4,
        semantic_feat_dim=4,
        axes_dims=(2, 2, 2),
        axes_lens=(256, 32, 32),
    ).eval()


def test_width_one_preserves_native_t2i_and_edit_outputs() -> None:
    torch.manual_seed(17)
    model = _tiny_model()
    x = [torch.randn(2, 1, 2, 2)]
    timestep = torch.tensor([0.6])
    cap_feats = [torch.randn(3, 4)]
    glm_feats = [torch.randn(5, 4)]
    source_latents = [torch.randn(2, 1, 2, 2)]

    with torch.inference_mode():
        native_t2i = model(
            x,
            timestep,
            cap_feats,
            glm_cap_feats=glm_feats,
            return_dict=False,
        )[0][0]
        native_edit = model(
            x,
            timestep,
            cap_feats,
            glm_cap_feats=glm_feats,
            source_latents=source_latents,
            return_dict=False,
        )[0][0]

    parallel = EpeParallelContext.from_torchrun(allowed_widths=(1,))
    try:
        model.configure_epe(parallel)
        with torch.inference_mode():
            wrapped_t2i = model(
                x,
                timestep,
                cap_feats,
                glm_cap_feats=glm_feats,
                return_dict=False,
            )[0][0]
            wrapped_edit = model(
                x,
                timestep,
                cap_feats,
                glm_cap_feats=glm_feats,
                source_latents=source_latents,
                return_dict=False,
            )[0][0]
    finally:
        parallel.close()

    assert torch.equal(wrapped_t2i, native_t2i)
    assert torch.equal(wrapped_edit, native_edit)


def test_llada_epe_warmup_uses_patchified_latent_shape() -> None:
    torch.manual_seed(23)
    model = _tiny_model(layers=0)
    parallel = EpeParallelContext.from_torchrun(allowed_widths=(1,))
    try:
        epe = LLaDAImageEpeModule(parallel, cfg_parallel=False)
        model.configure_epe(epe)
        report = epe.warmup_transformer(
            model,
            resolutions=((32, 32),),
            steps=3,
            vae_scale_factor=16,
            text_tokens=3,
            cfg_conditions=1,
        )
    finally:
        parallel.close()

    assert report["rows"][0]["raw_image_tokens"] == 4
    assert report["rows"][0]["image_tokens"] == 32
    assert report["rows"][0]["cp_degree"] == 1


def test_cp_attention_rejects_any_outer_mask_without_inspecting_values() -> None:
    processor = object.__new__(LLaDAImageCpAttnProcessor)
    processor._enabled = True

    with pytest.raises(NotImplementedError, match="outer ragged masks"):
        processor(
            None,
            torch.zeros(1, 1, 1),
            attention_mask=torch.ones(1, 1, dtype=torch.bool),
        )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(not rms_norm_available(), reason="Triton is required")
def test_llada_rms_norm_cuda_matches_diffusers_reference() -> None:
    torch.manual_seed(31)
    hidden_states = torch.randn(
        4,
        257,
        2560,
        device="cuda",
        dtype=torch.bfloat16,
    )
    reference = DiffusersRMSNorm(2560, eps=1e-5, elementwise_affine=False).cuda()
    optimized = LLaDAImageRMSNorm(2560, eps=1e-5, elementwise_affine=False).cuda()

    with torch.inference_mode():
        expected = reference(hidden_states)
        actual = optimized(hidden_states)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.008)
