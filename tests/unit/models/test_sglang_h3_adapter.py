from __future__ import annotations

import importlib.util
import inspect
import runpy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[3]
HAS_SGLANG = importlib.util.find_spec("sglang") is not None and (
    importlib.util.find_spec("sglang.multimodal_gen") is not None
)

if HAS_SGLANG:
    # Establish the same lightweight namespaces and process-local override used
    # by spawned SGLang workers.  h3-baseline intentionally omits unrelated
    # Chitu/Diffusers dependencies.
    runpy.run_path(
        str(ROOT / "script" / "launch_sglang_h3_chitu.py"),
        run_name="_chitu_sglang_h3_test_bootstrap",
    )

    from chitu_diffusion.integrations.sglang.bootstrap import (
        H3_ARCHITECTURE,
        install_sglang_h3_adapter,
        validate_launcher_args,
    )
    from chitu_diffusion.integrations.sglang.minimax_h3 import (
        _FORWARD_SUPPORTED_KWARGS,
        MiniMaxH3DiTModel,
    )
    from chitu_diffusion.models.minimax_h3.rope import MiniMaxH3Rope
    from sglang.multimodal_gen.runtime.models.registry import ModelRegistry


EXPECTED_FORWARD_KWARGS = {
    "x",
    "audio_x",
    "img_position_ids",
    "rope_cache",
    "unique_timesteps",
    "inverse_indices",
    "update_mask",
    "update_audio_mask",
    "token_tags",
    "block_token_tags",
    "block_combined_indices",
    "skip_mask_out_condition",
    "prompt_embeds",
    "refined_prompt_embeds_length",
    "img_pos_info",
    "audio_pos_info",
    "text_pos_info",
    "img_pos_for_infer_output_info",
    "local_embedding_layout",
    "packed_seq_params",
    "refiner_packed_seq_params",
}


def _tiny_sglang_config() -> SimpleNamespace:
    arch = SimpleNamespace(
        hidden_size=8,
        num_layers=1,
        token_refiner_num_layers=1,
        num_attention_heads=1,
        attention_head_dim=8,
        ffn_hidden_size=8,
        latents_dim=1,
        audio_latents_dim=1,
        patch_size=(1, 1, 1),
        text_dim=4,
        timestep_input_dim=4,
        time_embed_hidden_size=8,
        time_embed_dim=4,
        adaln_out_features=18 * 8,
        final_adaln_out_features=2 * 8,
        rope_inv_freq_len=1,
        norm_eps=1e-5,
        qk_norm_eps=1e-5,
        final_norm_eps=1e-5,
    )
    return SimpleNamespace(arch_config=arch)


@unittest.skipUnless(HAS_SGLANG, "requires sglang.multimodal_gen")
class TestSGLangH3Adapter(unittest.TestCase):
    def test_registry_override_resolves_adapter(self) -> None:
        install_sglang_h3_adapter()
        model_cls, architecture = ModelRegistry.resolve_model_cls(H3_ARCHITECTURE)
        self.assertEqual(architecture, H3_ARCHITECTURE)
        self.assertIs(model_cls, MiniMaxH3DiTModel)

    def test_constructor_contract_and_required_attrs_on_meta(self) -> None:
        signature = inspect.signature(MiniMaxH3DiTModel)
        self.assertEqual(
            tuple(signature.parameters),
            ("config", "hf_config", "quant_config"),
        )
        with torch.device("meta"):
            model = MiniMaxH3DiTModel(_tiny_sglang_config(), {}, None)
        self.assertEqual(model.hidden_size, 8)
        self.assertEqual(model.num_attention_heads, 1)
        self.assertEqual(model.num_channels_latents, 1)
        self.assertEqual(model.layer_names, ["blocks"])
        self.assertEqual(model.param_names_mapping, {})
        self.assertTrue(callable(model.blocks[0].attn.qkv_proj.weight.weight_loader))
        model.post_load_weights()
        with self.assertRaisesRegex(NotImplementedError, "FSDP"):
            _ = model._fsdp_shard_conditions
        with self.assertRaisesRegex(NotImplementedError, "quantized"):
            MiniMaxH3DiTModel(_tiny_sglang_config(), {}, object())

    def test_forward_kwarg_contract_is_exhaustive_and_strict(self) -> None:
        self.assertEqual(_FORWARD_SUPPORTED_KWARGS, EXPECTED_FORWARD_KWARGS)
        model = MiniMaxH3DiTModel.__new__(MiniMaxH3DiTModel)
        nn.Module.__init__(model)
        with self.assertRaisesRegex(TypeError, "unexpected kwargs"):
            model(unrecognized_tensor=torch.empty(0))
        with self.assertRaisesRegex(ValueError, "'x'"):
            model()
        with self.assertRaisesRegex(NotImplementedError, "breakable CUDA graph"):
            model(block_token_tags=torch.empty(0))

    def test_build_rope_cache_shape_contract(self) -> None:
        model = MiniMaxH3DiTModel.__new__(MiniMaxH3DiTModel)
        nn.Module.__init__(model)
        model.rope = MiniMaxH3Rope(2)
        position_ids = torch.tensor(
            [[[0, 0, 0], [1, 2, 3], [2, 3, 4], [3, 4, 5]]],
            dtype=torch.long,
        )
        cache, positions = model.build_rope_cache(
            position_ids,
            device=torch.device("cpu"),
        )
        self.assertEqual(cache.shape, (4, 12))
        self.assertEqual(cache.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(positions, torch.arange(4)))

    def test_launcher_rejects_unsupported_modes(self) -> None:
        with self.assertRaisesRegex(ValueError, "FSDP"):
            validate_launcher_args(["serve", "--use-fsdp-inference"])
        with self.assertRaisesRegex(ValueError, "breakable CUDA"):
            validate_launcher_args(["serve", "--enable-breakable-cuda-graph"])
        with self.assertRaisesRegex(ValueError, "Ref2VA"):
            validate_launcher_args(["serve", "--model-variant", "ref2va"])
        validate_launcher_args(
            [
                "serve",
                "--model-variant",
                "fl2va",
                "--use-fsdp-inference",
                "false",
            ]
        )

    def test_does_not_create_groups_or_epac_context(self) -> None:
        source = inspect.getsource(sys.modules[MiniMaxH3DiTModel.__module__])
        self.assertNotIn("new_group(", source)
        self.assertNotIn("EpeParallelContext", source)


if __name__ == "__main__":
    unittest.main()

