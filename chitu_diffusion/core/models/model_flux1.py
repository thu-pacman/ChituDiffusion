# Copyright 2024 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FluxTransformer2DLoadersMixin, FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import (
    Attention,
    AttentionProcessor,
    FluxAttnProcessor2_0_NPU,
    FusedFluxAttnProcessor2_0,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.embeddings import CombinedTimestepGuidanceTextProjEmbeddings, CombinedTimestepTextProjEmbeddings, FluxPosEmbed
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from chitu_diffusion.core.models.backbone import BackboneBlockInfo, BackboneMixin, BackboneState
from chitu_diffusion.core.models.registry import ModelType, register_model
from chitu_diffusion.modules.attention.flux_attention import ChituFluxAttnProcessor2_0, FluxAttnProcessor2_0

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@maybe_allow_in_graph
class FluxSingleTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=4.0, attn_backend=None):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)

        if attn_backend is not None:
            processor = ChituFluxAttnProcessor2_0(attn_backend)
        elif is_torch_npu_available():
            processor = FluxAttnProcessor2_0_NPU()
        else:
            processor = FluxAttnProcessor2_0()
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=processor,
            qk_norm="rms_norm",
            eps=1e-6,
            pre_only=True,
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
    ):
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        joint_attention_kwargs = joint_attention_kwargs or {}
        
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states


@maybe_allow_in_graph
class FluxTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        qk_norm="rms_norm",
        eps=1e-6,
        attn_backend=None,
    ):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim)

        self.norm1_context = AdaLayerNormZero(dim)

        if attn_backend is not None:
            processor = ChituFluxAttnProcessor2_0(attn_backend)
        elif hasattr(F, "scaled_dot_product_attention"):
            processor = FluxAttnProcessor2_0()
        else:
            raise ValueError(
                "The current PyTorch version does not support the `scaled_dot_product_attention` function."
            )
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=processor,
            qk_norm=qk_norm,
            eps=eps,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )
        joint_attention_kwargs = joint_attention_kwargs or {}
        # Attention.
        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
        elif len(attention_outputs) == 3:
            attn_output, context_attn_output, ip_attn_output = attention_outputs

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output
        if len(attention_outputs) == 3:
            hidden_states = hidden_states + ip_attn_output

        # Process attention outputs for the `encoder_hidden_states`.

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


@register_model(ModelType.FLUX1_DEV)
class Flux1Model(
    BackboneMixin, ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, FluxTransformer2DLoadersMixin
):
    """
    The Transformer model introduced in Flux.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Parameters:
        patch_size (`int`): Patch size to turn the input data into small patches.
        in_channels (`int`, *optional*, defaults to 16): The number of channels in the input.
        num_layers (`int`, *optional*, defaults to 18): The number of layers of MMDiT blocks to use.
        num_single_layers (`int`, *optional*, defaults to 18): The number of layers of single DiT blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 18): The number of heads to use for multi-head attention.
        joint_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        pooled_projection_dim (`int`): Number of dimensions to use when projecting the `pooled_projections`.
        guidance_embeds (`bool`, defaults to False): Whether to use guidance embeddings.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        model_type: str = "t2i",
        attn_backend=None,
        rope_impl=None,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: Optional[int] = None,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int] = (16, 56, 56),
    ):
        super().__init__()
        if model_type != "t2i":
            raise ValueError(f"Flux.1 only supports t2i, got {model_type}.")
        self.out_channels = out_channels or in_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)

        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings if guidance_embeds else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
        )

        self.context_embedder = nn.Linear(self.config.joint_attention_dim, self.inner_dim)
        self.x_embedder = nn.Linear(self.config.in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    attn_backend=attn_backend,
                )
                for i in range(self.config.num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    attn_backend=attn_backend,
                )
                for i in range(self.config.num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False

        # RoPE cos/sin only depend on the position `ids`, which are identical across
        # all denoise steps of a generation. Memoize the last (ids -> cos/sin) so we
        # skip the per-step float64 pos_embed recompute (also keeps the float64 work
        # out of any compiled region).
        self._rope_cache_ids = None
        self._rope_cache_val = None
        # Set by the runtime when compile_scope == "model": a torch.compile-wrapped
        # `_dit_core`. When present (and no controlnet), forward routes through it so
        # the whole block stack is one compiled graph (compatible with step-level
        # FlexCache that decides replay/skip outside the model forward).
        self._compiled_core = None

    def get_teacache_modulated_input(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        modulated_inp, *_ = self.transformer_blocks[0].norm1(hidden_states, emb=temb)
        return modulated_inp

    def backbone_blocks(self):
        blocks = []
        for index, block in enumerate(self.transformer_blocks):
            blocks.append(BackboneBlockInfo(index=index, name=f"transformer_blocks.{index}", module=block))
        offset = len(blocks)
        for index, block in enumerate(self.single_transformer_blocks):
            blocks.append(
                BackboneBlockInfo(
                    index=offset + index,
                    name=f"single_transformer_blocks.{index}",
                    module=block,
                )
            )
        return blocks

    def backbone_attention_modules(self):
        modules = []
        for block_info in self.backbone_blocks():
            if hasattr(block_info.module, "attn"):
                modules.append((block_info.index, "self", block_info.module.attn))
        return modules

    def backbone_make_state(self, hidden_states: torch.Tensor, **kwargs):
        kwargs = dict(kwargs)
        kwargs.pop("raw_e", None)
        encoder_hidden_states = kwargs["encoder_hidden_states"]
        return BackboneState({
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "text_seq_len": encoder_hidden_states.shape[1],
            "temb": kwargs["temb"],
            "image_rotary_emb": kwargs["image_rotary_emb"],
            "joint_attention_kwargs": kwargs.get("joint_attention_kwargs"),
            "controlnet_block_samples": kwargs.get("controlnet_block_samples"),
            "controlnet_single_block_samples": kwargs.get("controlnet_single_block_samples"),
            "controlnet_blocks_repeat": kwargs.get("controlnet_blocks_repeat", False),
            "single_stream_started": False,
        })

    def backbone_prepare_block_state(self, block_info: BackboneBlockInfo, state):
        if block_info.index >= len(self.transformer_blocks) and not state["single_stream_started"]:
            state = BackboneState(state)
            state["hidden_states"] = torch.cat([state["encoder_hidden_states"], state["hidden_states"]], dim=1)
            state["single_stream_started"] = True
        return state

    def block_compute(self, block_info: BackboneBlockInfo, state):
        block_index = block_info.index
        if block_index < len(self.transformer_blocks):
            block = self.transformer_blocks[block_index]
            encoder_hidden_states, hidden_states = block(
                hidden_states=state["hidden_states"],
                encoder_hidden_states=state["encoder_hidden_states"],
                temb=state["temb"],
                image_rotary_emb=state["image_rotary_emb"],
                joint_attention_kwargs=state["joint_attention_kwargs"],
            )

            controlnet_block_samples = state["controlnet_block_samples"]
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                if state["controlnet_blocks_repeat"]:
                    hidden_states = hidden_states + controlnet_block_samples[block_index % len(controlnet_block_samples)]
                else:
                    hidden_states = hidden_states + controlnet_block_samples[block_index // interval_control]

            state["hidden_states"] = hidden_states
            state["encoder_hidden_states"] = encoder_hidden_states
            return state

        single_index = block_index - len(self.transformer_blocks)
        block = self.single_transformer_blocks[single_index]
        hidden_states = block(
            hidden_states=state["hidden_states"],
            temb=state["temb"],
            image_rotary_emb=state["image_rotary_emb"],
            joint_attention_kwargs=state["joint_attention_kwargs"],
        )

        controlnet_single_block_samples = state["controlnet_single_block_samples"]
        if controlnet_single_block_samples is not None:
            interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
            interval_control = int(np.ceil(interval_control))
            text_seq_len = state["text_seq_len"]
            hidden_states[:, text_seq_len:, ...] = (
                hidden_states[:, text_seq_len:, ...]
                + controlnet_single_block_samples[single_index // interval_control]
            )
        state["hidden_states"] = hidden_states
        return state

    def backbone_run_block(self, block_info: BackboneBlockInfo, state):
        return self.block_compute(block_info, state)

    def backbone_state_tensor(self, state) -> torch.Tensor:
        if state["single_stream_started"]:
            return state["hidden_states"]
        return state["hidden_states"]

    def backbone_with_state_tensor(self, state, tensor: torch.Tensor):
        state = BackboneState(state)
        state["hidden_states"] = tensor
        if tensor.shape[1] > state["text_seq_len"]:
            state["single_stream_started"] = True
        return state

    def backbone_finalize_state(self, state) -> torch.Tensor:
        if state["single_stream_started"]:
            return state["hidden_states"][:, state["text_seq_len"] :, ...]
        return state["hidden_states"]

    def backbone_cache_state(self, state):
        if state["single_stream_started"]:
            return {"hidden_states": state["hidden_states"].detach()}
        return {
            "hidden_states": state["hidden_states"].detach(),
            "encoder_hidden_states": state["encoder_hidden_states"].detach(),
        }

    def backbone_restore_cached_state(self, state, cached_state):
        state = BackboneState(state)
        state["hidden_states"] = cached_state["hidden_states"]
        if "encoder_hidden_states" in cached_state:
            state["encoder_hidden_states"] = cached_state["encoder_hidden_states"]
            state["single_stream_started"] = False
        else:
            state["single_stream_started"] = True
        return state

    def backbone_block_delta(self, before, after):
        delta = {"hidden_states": after["hidden_states"] - before["hidden_states"]}
        if "encoder_hidden_states" in after:
            delta["encoder_hidden_states"] = after["encoder_hidden_states"] - before["encoder_hidden_states"]
        return delta

    def backbone_apply_block_delta(self, state, delta):
        state = BackboneState(state)
        state["hidden_states"] = state["hidden_states"] + delta["hidden_states"]
        if "encoder_hidden_states" in delta:
            state["encoder_hidden_states"] = state["encoder_hidden_states"] + delta["encoder_hidden_states"]
        return state

    def _cached_rope(self, ids: torch.Tensor):
        """Return RoPE (cos, sin) for `ids`, memoizing the last result.

        Positions are constant across denoise steps, so this turns the per-step
        float64 pos_embed recompute into a cheap tensor-equality check on hits.
        """
        cached_ids = self._rope_cache_ids
        if (
            cached_ids is not None
            and cached_ids.shape == ids.shape
            and cached_ids.device == ids.device
            and torch.equal(cached_ids, ids)
        ):
            return self._rope_cache_val
        val = self.pos_embed(ids)
        self._rope_cache_ids = ids
        self._rope_cache_val = val
        return val

    def _dit_core(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb,
        text_seq_len: int,
    ) -> torch.Tensor:
        """Pure-tensor DiT block stack (joint stream then single stream).

        Inlines the block loop without BackboneState/BackboneBlockInfo so the whole
        stack traces cleanly as a single graph for model-scope torch.compile. Does
        not support controlnet / joint_attention_kwargs; the caller falls back to
        `model_compute` when those are present.
        """
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        for block in self.single_transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )
        return hidden_states[:, text_seq_len:, ...]

    def model_compute(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        controlnet_blocks_repeat: bool = False,
    ) -> torch.Tensor:
        state = self.backbone_make_state(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
            controlnet_block_samples=controlnet_block_samples,
            controlnet_single_block_samples=controlnet_single_block_samples,
            controlnet_blocks_repeat=controlnet_blocks_repeat,
        )
        for block_info in self.backbone_blocks():
            state = self.backbone_prepare_block_state(block_info, state)
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                state = torch.utils.checkpoint.checkpoint(
                    lambda current_state: self.block_compute(block_info, current_state),
                    state,
                    **ckpt_kwargs,
                )
            else:
                state = self.block_compute(block_info, state)
        return self.backbone_finalize_state(state)

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections with FusedAttnProcessor2_0->FusedFluxAttnProcessor2_0
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedFluxAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self._cached_rope(ids)

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        # Model-scope compile fast path: one compiled graph over the whole block
        # stack. Only valid without controlnet / extra joint-attention kwargs.
        use_compiled_core = (
            self._compiled_core is not None
            and controlnet_block_samples is None
            and controlnet_single_block_samples is None
            and not joint_attention_kwargs
        )
        if use_compiled_core:
            hidden_states = self._compiled_core(
                hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb,
                encoder_hidden_states.shape[1],
            )
        else:
            hidden_states = self.model_compute(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
                controlnet_block_samples=controlnet_block_samples,
                controlnet_single_block_samples=controlnet_single_block_samples,
                controlnet_blocks_repeat=controlnet_blocks_repeat,
            )

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
