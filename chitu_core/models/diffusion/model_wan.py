# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import itertools
import math
from logging import getLogger
from typing import Optional

import torch
import torch.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from chitu_core.models.registry import ModelType, register_model, log_init_params
from chitu_core.distributed.partition import compute_layer_dist_in_pp
from chitu_core.distributed.parallel_state import get_fpp_group  
from chitu_diffusion.model_default import WanModelDefaults
from chitu_diffusion.modules.attention.wan_attention import flash_attention


from chitu_diffusion.flex_cache.flexcache_manager import FlexCacheManager
# from chitu_diffusion.flex_cache.strategy.FPPCache import FPPCache


logger = getLogger(__name__)

__all__ = ['WanModel']

T5_CONTEXT_TOKEN_NUMBER = 512
FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER = 257 * 2


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast(device_type="cuda", enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs



class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)

def half(x):
    return x if x.dtype in (torch.float16, torch.bfloat16) else x.to(torch.bfloat16)

@amp.autocast(device_type="cuda", enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()

class WanSelfAttention(nn.Module):

    def __init__(self,
                 attn_func,
                 rope_impl,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.attn_func = attn_func
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.rope_impl = rope_impl or rope_apply


    def forward(self, x, grid_sizes, freqs, save_cache=False, position_idx=None, cache_manager = None, layer_idx = None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if position_idx is not None:
            rope_q = self.rope_impl(q, grid_sizes, freqs)
            rope_k = self.rope_impl(k, grid_sizes, freqs)
        else:
            from chitu_diffusion.modules.rope.diffusion_rope_backend import naive_rope_apply
            rope_q = naive_rope_apply(q, grid_sizes, freqs)
            rope_k = naive_rope_apply(k, grid_sizes, freqs)
        ## should save cache here
        if save_cache:
            from chitu_diffusion.flex_cache.strategy.FPPCache import FPPCache
            cache_manager : FPPCache
            if position_idx is None:
                cache_manager.init_layer_stale_kv(rope_k, v, layer_idx)
            else:
                rope_k, v = cache_manager.update_layer_stale_kv_patch(rope_k, v, layer_idx, (position_idx, position_idx + s))
            


        x = self.attn_func(
            q = half(rope_q),
            k = half(rope_k),
            v = half(v),
            window_size=self.window_size,

        )[0]
        x = x.to(q.dtype)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)
        
        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        image_context_length = context.shape[1] - T5_CONTEXT_TOKEN_NUMBER
        context_img = context[:, :image_context_length]
        context = context[:, image_context_length:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}

class WanAttentionBlock(nn.Module):

    def __init__(self,
                 attn_func,
                 rope_impl,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(attn_func, rope_impl, 
                                          dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](attn_func,
                                                                      rope_impl,
                                                                      dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        context,
        grid_sizes,
        freqs,
        context_lens=None,
        save_cache=False,
        position_idx=None,
        cache_manager = None,
        layer_idx = None
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        with amp.autocast(device_type="cuda", dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1]) + e[0], grid_sizes,
            freqs, save_cache, position_idx, cache_manager, layer_idx)
        
        with amp.autocast(device_type="cuda", dtype=torch.float32):
            x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
            with amp.autocast(device_type="cuda", dtype=torch.float32):
                x = x + y * e[5]
            return x
        
        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        assert e.dtype == torch.float32
        with amp.autocast(device_type="cuda", dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim, flf_pos_emb=False):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))
        if flf_pos_emb:  # NOTE: we only use this for `flf2v`
            self.emb_pos = nn.Parameter(
                torch.zeros(1, FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER, 1280))

    def forward(self, image_embeds):
        if hasattr(self, 'emb_pos'):
            bs, n, d = image_embeds.shape
            image_embeds = image_embeds.view(-1, 2 * n, d)
            image_embeds = image_embeds + self.emb_pos
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens

@register_model(ModelType.WAN_DIT)
@log_init_params
class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    def __init__(self, model_type='t2v', attn_backend=None, rope_impl=None, **kwargs):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video) or 'flf2v' (first-last-frame-to-video) or 'vace'
            **kwargs: 其他超参数，如果未提供则使用默认值
        """

        super().__init__()

        # logger.info(f"Initializing WanModel with type: {model_type}")

        assert model_type in ['t2v', 'i2v', 'flf2v', 'vace']
        self.model_type = model_type

        # 使用默认值填充缺失的参数
        defaults = WanModelDefaults()
        
        self.patch_size = kwargs.get('patch_size', defaults.patch_size)
        self.text_len = kwargs.get('text_len', defaults.text_len)
        self.in_dim = kwargs.get('in_dim', defaults.in_dim)
        self.dim = kwargs.get('dim', defaults.dim)
        self.ffn_dim = kwargs.get('ffn_dim', defaults.ffn_dim)
        self.freq_dim = kwargs.get('freq_dim', defaults.freq_dim)
        self.text_dim = kwargs.get('text_dim', defaults.text_dim)
        self.out_dim = kwargs.get('out_dim', defaults.out_dim)
        self.num_heads = kwargs.get('num_heads', defaults.num_heads)
        self.num_layers = kwargs.get('num_layers', defaults.num_layers)
        self.window_size = kwargs.get('window_size', defaults.window_size)
        self.qk_norm = kwargs.get('qk_norm', defaults.qk_norm)
        self.cross_attn_norm = kwargs.get('cross_attn_norm', defaults.cross_attn_norm)
        self.eps = kwargs.get('eps', defaults.eps)

        # embeddings
        self.patch_embedding = nn.Conv3d(
            self.in_dim, self.dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(self.text_dim, self.dim), nn.GELU(approximate='tanh'),
            nn.Linear(self.dim, self.dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.dim), nn.SiLU(), nn.Linear(self.dim, self.dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(self.dim, self.dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(attn_backend, rope_impl, 
                              cross_attn_type, self.dim, self.ffn_dim, self.num_heads,
                              self.window_size, self.qk_norm, self.cross_attn_norm, self.eps)
            for _ in range(self.num_layers)
        ])

        # head
        self.head = Head(self.dim, self.out_dim, self.patch_size, self.eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (self.dim % self.num_heads) == 0 and (self.dim // self.num_heads) % 2 == 0
        self._freqs = None

        if model_type == 'i2v' or model_type == 'flf2v':
            self.img_emb = MLPProj(1280, self.dim, flf_pos_emb=model_type == 'flf2v')

        self.fpp_size = get_fpp_group().group_size
        # initialize weights
        self.init_weights()

        self.cache_manager = None
    
    # 这个函数会在模型初始化后被调用，用于根据当前的pp_rank调整模型层的分布    
    def wrap_layers_for_fpp(self):
        self.pp_rank = get_fpp_group().rank_in_group
        self.fpp_group = get_fpp_group()
        self.counter = 0

        print(f"Initializing layers for pp_rank {self.pp_rank} / fpp_size {self.fpp_size}")

        num_layers_of_each_rank = compute_layer_dist_in_pp(
            self.num_layers, self.fpp_size
        )
        first_layer_id_of_each_rank = list(
            itertools.accumulate([0] + num_layers_of_each_rank)
        )
        self.local_begin_layer_id = first_layer_id_of_each_rank[self.pp_rank]
        self.local_end_layer_id = first_layer_id_of_each_rank[self.pp_rank + 1]

        self.blocks = self.blocks[self.local_begin_layer_id:self.local_end_layer_id]

    @property
    def freqs(self):
        """
        Delay initializing `self.freqs` to make sure it is not on meta device
        """
        if self._freqs is None or self._freqs.is_meta:
            # real current device
            device = self.patch_embedding.weight.device
            if device.type == 'meta':
                return self._freqs # return if still meta device (should not happen)            
            d = self.dim // self.num_heads
            self._freqs = torch.cat([
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6))
            ], dim=1).to(device=device)
            
        return self._freqs

    def log(self, msg):
        # 分布式调试用
        # 封装日志函数，统一加时间戳+rank+flush
        import time
        print(f"[{time.strftime('%H:%M:%S.%f')}] Rank {self.pp_rank}: {msg}", flush=True)

    def model_compute(self, tokens, time_proj, context_embedding, grid_sizes,  context_lens=None, save_cache=False, position_idx=None):
        """
        主计算负载：通过所有transformer blocks处理tokens
        这是计算密集的核心部分，适合分布式处理
        
        Args:
            tokens: 输入tokens shape [B, L(padded), dim`]
            time_proj: 时间投影
            context_embedding: 上下文嵌入
            grid_sizes: 网格大小
            context_lens: 上下文长度

        Returns:
            processed_tokens: 处理后的tokens
        """

        x = tokens
        for i, block in enumerate(self.blocks):
            # print(f"model_compute: cache_manager = {self.cache_manager}, strategy = {self.cache_manager.strategy}, layer_idx={i}")
            x = block(x, time_proj, context_embedding, grid_sizes, self.freqs, context_lens, save_cache=save_cache, position_idx=position_idx, cache_manager=self.cache_manager.strategy, layer_idx=i)
        return x

    
    

    def _cal_patch_embedding(self, x, seq_len, y=None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        """
        # 阶段1: latents -> tokens
        if not isinstance(x, list):
            x = [x]

        if self.model_type == 'i2v' or self.model_type == 'flf2v':
            assert y is not None

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings - 将latents转换为patch embeddings
        # x : list[C(in_dim), F, H, W] -> list[1, C_o(dim), F, H, W]
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        # x : list[1, C_o(dim), F_patches, H_patches, W_patches] -> list[1, F*H*W, dim]
        x = [u.flatten(2).transpose(1, 2) for u in x]
        # 创建tokens - 这是将要传递给主计算的数据
        # pad tokens to be divisible by split_size (cp_size * fpp_size)

        tokens = torch.cat(x)

        return tokens
    
    def _cal_grid_sizes(self, latent_shape: torch.Size):

        # grid_sizes = torch.tensor(latent_shape[1:] // self.patch_size, dtype=torch.long)

        grid_sizes = torch.tensor([
            latent_shape[1] // self.patch_size[0], 
            latent_shape[2] // self.patch_size[1], 
            latent_shape[3] // self.patch_size[2]], 
            dtype=torch.long).reshape(1, 3)
        return grid_sizes

    def _cal_timeproj(self, t):

        t = t.unsqueeze_(0) if t.dim() == 0 else t
    
        with amp.autocast(device_type="cuda", dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        return e0
    
    def _cal_time_embeddings(self, t):
        t = t.unsqueeze(0) if t.dim() == 0 else t
        with amp.autocast(device_type="cuda", dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
        return e

    def _cal_context_embeddings(self, context, clip_fea=None):
        if not isinstance(context, list):
            context = [context]

        if self.model_type == 'i2v' or self.model_type == 'flf2v':
            assert clip_fea is not None

        with amp.autocast(device_type="cuda", dtype=torch.float32):
            context = self.text_embedding(
                torch.stack([
                    torch.cat(
                        [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in context
                ]))
        
        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 (x2) x dim
            context = torch.concat([context_clip, context], dim=1)

        return context
    

    def _post_dit(self, x, e, grid_sizes):
        # head processing
        x = self.head(x, e)
        # unpatchify - 将tokens转换回空间表示
        x = self.unpatchify(x, grid_sizes)
        return x  
            
    def forward(
        self,
        latent,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
    ):
        """
        完整的前向传播，现在拆分为三个阶段
        x: latents, [C, F_l, H_l, W_l] or List[Latent]
        """
        tokens = self._cal_patch_embedding(latent, seq_len, y)
        grid_sizes = self._cal_grid_sizes(latent.shape)
        time_proj = self._cal_timeproj(t)
        context_embedding = self._cal_context_embeddings(context, clip_fea)


        tokens = self.model_compute(tokens, time_proj, context_embedding, grid_sizes)

        time_embedding = self._cal_time_embeddings(t)
        latent = self._post_dit(tokens, time_embedding, grid_sizes)

        return latent[0].to(torch.float32)


    
    def sync_pipe_forward(self, latent, t, context, seq_len, clip_fea=None, y=None, save_cache=False):
        """
        同步管道前向传播：在每个阶段之间进行同步，适用于分布式环境
        """
        tokens = self._cal_patch_embedding(latent, seq_len, y)
        grid_sizes = self._cal_grid_sizes(latent.shape)  # [1,3]
        time_proj = self._cal_timeproj(t)
        context_embedding = self._cal_context_embeddings(context, clip_fea)
        
        if not self.fpp_group.is_first_rank: 
            
            tokens = self.fpp_group.p2p_irecv(tokens.shape, torch.float32, self.fpp_group.prev_rank,tag=5)
            self.fpp_group.p2p_commit()
            self.fpp_group.p2p_wait()

        tokens = self.model_compute(tokens, time_proj, context_embedding, grid_sizes, save_cache=save_cache)

        if not self.fpp_group.is_last_rank:
            # print(f"dtype before send: {tokens.dtype}")

            self.fpp_group.p2p_isend(tokens.to(torch.float32), self.fpp_group.next_rank,tag=5)
            self.fpp_group.p2p_commit()
            self.fpp_group.p2p_wait()
            return None 

        if self.fpp_group.is_last_rank:
            from chitu_diffusion.flex_cache.strategy.FPPCache import FPPCache
            cache_strategy : FPPCache = self.cache_manager.strategy 
            cache_strategy.init_stale_tokens(tokens)
            # print(f"Rank {self.pp_rank}: saved_stale_tokens", flush=True)

            time_embedding = self._cal_time_embeddings(t)
            latents = self._post_dit(tokens, time_embedding, grid_sizes)
            return latents[0].to(torch.float32)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
