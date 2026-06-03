from typing import Dict, Tuple

import torch
import torch.amp as amp

from chitu_diffusion.core.models.model_wan import sinusoidal_embedding_1d
from chitu_diffusion.modules.attention.wan_attention import flash_attention


class WanCubicSelectiveForwardEngine:
    """Cubic-WAN forward adapter with same-source full and selective paths."""

    def __init__(self):
        self.k_cache: Dict[Tuple[str, int], torch.Tensor] = {}
        self.v_cache: Dict[Tuple[str, int], torch.Tensor] = {}
        self.residual_cache: Dict[str, torch.Tensor] = {}
        self.last_forward_mode = "uninitialized"

    def reset(self):
        self.k_cache.clear()
        self.v_cache.clear()
        self.residual_cache.clear()
        self.last_forward_mode = "reset"

    def forward(
        self,
        module,
        original_forward,
        x,
        t,
        context,
        seq_len,
        branch_key: str,
        step_plan=None,
        clip_fea=None,
        y=None,
    ):
        if getattr(module, "model_type", None) != "t2v":
            self.last_forward_mode = "original_forward"
            return original_forward(x, t=t, context=context, seq_len=seq_len, clip_fea=clip_fea, y=y)
        if not isinstance(x, list):
            x, t, context, y = module._single_input_preprocess(x, t, context, y)
        if len(x) != 1:
            raise ValueError("Cubic WAN selective forward currently supports batch size 1 only.")
        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        x_emb = [module.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x_emb])
        x_seq = [u.flatten(2).transpose(1, 2) for u in x_emb]
        seq_lens = torch.tensor([u.size(1) for u in x_seq], dtype=torch.long)
        if seq_lens.max() > seq_len:
            raise ValueError(f"Cubic WAN seq_len {int(seq_len)} is smaller than actual {int(seq_lens.max())}.")
        x_full = x_seq[0]
        total_tokens = int(seq_lens[0].item())

        with amp.autocast(device_type="cuda", dtype=torch.float32):
            e = module.time_embedding(sinusoidal_embedding_1d(module.freq_dim, t).float())
            e0 = module.time_projection(e).unflatten(1, (6, module.dim))

        context_lens = None
        context = module.text_embedding(
            torch.stack([
                torch.cat([u, u.new_zeros(module.text_len - u.size(0), u.size(1))])
                for u in context
            ])
        )
        if clip_fea is not None:
            context_clip = module.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1)

        use_selective = step_plan is not None and (not step_plan.is_full_compute)
        active_idx = None
        if use_selective:
            active_idx = step_plan.active_token_indices.long().to(x_full.device)
            if int(active_idx.numel()) >= total_tokens:
                use_selective = False

        if not use_selective:
            self.last_forward_mode = "full_refresh"
            return self._full_forward_and_refresh_cache(
                module=module,
                x_full=x_full,
                e=e,
                e0=e0,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                context=context,
                context_lens=context_lens,
                branch_key=branch_key,
            )

        if int(active_idx.numel()) == 0:
            out = self._cached_residual_forward(module, x_full, e, grid_sizes, branch_key)
            if out is not None:
                self.last_forward_mode = "cached_residual"
                return out
            self.last_forward_mode = "full_refresh"
            return self._full_forward_and_refresh_cache(
                module, x_full, e, e0, seq_lens, grid_sizes, context, context_lens, branch_key
            )

        if not self._has_cache(module, branch_key):
            self.last_forward_mode = "full_refresh"
            return self._full_forward_and_refresh_cache(
                module, x_full, e, e0, seq_lens, grid_sizes, context, context_lens, branch_key
            )

        kv_cache_token_indices = getattr(step_plan, "kv_cache_token_indices", step_plan.active_token_indices)
        kv_cache_update_positions = self._active_positions_for_tokens(
            active_idx=active_idx,
            token_indices=kv_cache_token_indices.long().to(x_full.device),
        )
        x_active = x_full.index_select(1, active_idx)
        for layer_idx, block in enumerate(module.blocks):
            x_active = self._selective_block_forward(
                block=block,
                x_active=x_active,
                e_time=e0,
                active_idx=active_idx,
                total_tokens=total_tokens,
                grid_size=grid_sizes[0],
                freqs=module.freqs,
                context=context,
                context_lens=context_lens,
                branch_key=branch_key,
                layer_idx=layer_idx,
                kv_cache_update_positions=kv_cache_update_positions,
            )

        residual_cache = self.residual_cache.get(branch_key)
        if residual_cache is None or residual_cache.size(0) != x_full.size(1):
            self.last_forward_mode = "full_refresh"
            return self._full_forward_and_refresh_cache(
                module, x_full, e, e0, seq_lens, grid_sizes, context, context_lens, branch_key
            )

        hidden_full = x_full[0] + residual_cache.to(x_full.device)
        hidden_full[active_idx] = x_active[0]
        updated_residual = residual_cache.clone()
        updated_residual[active_idx] = x_active[0] - x_full[0, active_idx]
        self.residual_cache[branch_key] = updated_residual.detach()

        out_full = module.head(hidden_full.unsqueeze(0), e)
        denoised = module.unpatchify(out_full, grid_sizes)
        self.last_forward_mode = "selective"
        return denoised[0].to(torch.float32)

    def _full_forward_and_refresh_cache(
        self,
        module,
        x_full,
        e,
        e0,
        seq_lens,
        grid_sizes,
        context,
        context_lens,
        branch_key: str,
    ):
        total_tokens = int(seq_lens[0].item())
        all_idx = torch.arange(total_tokens, device=x_full.device, dtype=torch.long)
        x_active = x_full
        for layer_idx, block in enumerate(module.blocks):
            x_active = self._selective_block_forward(
                block=block,
                x_active=x_active,
                e_time=e0,
                active_idx=all_idx,
                total_tokens=total_tokens,
                grid_size=grid_sizes[0],
                freqs=module.freqs,
                context=context,
                context_lens=context_lens,
                branch_key=branch_key,
                layer_idx=layer_idx,
            )
        self.residual_cache[branch_key] = (x_active[0] - x_full[0]).detach()
        out_full = module.head(x_active, e)
        denoised = module.unpatchify(out_full, grid_sizes)
        return denoised[0].to(torch.float32)

    def _cached_residual_forward(self, module, x_full, e, grid_sizes, branch_key: str):
        residual_cache = self.residual_cache.get(branch_key)
        if residual_cache is None or residual_cache.size(0) != x_full.size(1):
            return None
        hidden_full = x_full[0] + residual_cache.to(x_full.device)
        out_full = module.head(hidden_full.unsqueeze(0), e)
        denoised = module.unpatchify(out_full, grid_sizes)
        return denoised[0].to(torch.float32)

    def _selective_block_forward(
        self,
        block,
        x_active,
        e_time,
        active_idx,
        total_tokens: int,
        grid_size,
        freqs,
        context,
        context_lens,
        branch_key: str,
        layer_idx: int,
        kv_cache_update_positions=None,
    ):
        with amp.autocast(device_type="cuda", dtype=torch.float32):
            e0, e1, e2, e3, e4, e5 = (block.modulation + e_time).chunk(6, dim=1)

        x_for_attn = block.norm1(x_active).float() * (1 + e1) + e0
        bsz, active_len, _ = x_for_attn.shape
        num_heads = block.self_attn.num_heads
        head_dim = block.self_attn.head_dim

        q = block.self_attn.norm_q(block.self_attn.q(x_for_attn)).view(bsz, active_len, num_heads, head_dim)
        k_active = block.self_attn.norm_k(block.self_attn.k(x_for_attn)).view(bsz, active_len, num_heads, head_dim)
        v_active = block.self_attn.v(x_for_attn).view(bsz, active_len, num_heads, head_dim)

        q = self._rope_apply_indexed(q, active_idx, grid_size, freqs)
        k_active = self._rope_apply_indexed(k_active, active_idx, grid_size, freqs)

        k_key = (branch_key, layer_idx)
        v_key = (branch_key, layer_idx)
        k_cache = self.k_cache.get(k_key)
        v_cache = self.v_cache.get(v_key)
        if k_cache is None or v_cache is None or k_cache.size(0) != total_tokens:
            k_cache = q.new_zeros((total_tokens, num_heads, head_dim))
            v_cache = v_active.new_zeros((total_tokens, num_heads, head_dim))

        k_cache = k_cache.to(q.device)
        v_cache = v_cache.to(v_active.device)
        if kv_cache_update_positions is None:
            kv_cache_update_positions = torch.arange(active_len, device=active_idx.device, dtype=torch.long)
        else:
            kv_cache_update_positions = kv_cache_update_positions.to(active_idx.device)

        cache_update_count = int(kv_cache_update_positions.numel())
        if cache_update_count == active_len:
            k_cache[active_idx] = k_active[0]
            v_cache[active_idx] = v_active[0]
            self.k_cache[k_key] = k_cache.detach()
            self.v_cache[v_key] = v_cache.detach()
            k_full = k_cache.unsqueeze(0)
            v_full = v_cache.unsqueeze(0)
        else:
            if cache_update_count > 0:
                cache_update_idx = active_idx.index_select(0, kv_cache_update_positions)
                k_cache[cache_update_idx] = k_active[0].index_select(0, kv_cache_update_positions)
                v_cache[cache_update_idx] = v_active[0].index_select(0, kv_cache_update_positions)
                self.k_cache[k_key] = k_cache.detach()
                self.v_cache[v_key] = v_cache.detach()
            k_full = k_cache.unsqueeze(0).clone()
            v_full = v_cache.unsqueeze(0).clone()
            k_full[0, active_idx] = k_active[0]
            v_full[0, active_idx] = v_active[0]

        q_lens = torch.tensor([active_len], dtype=torch.long, device=q.device)
        k_lens = torch.tensor([total_tokens], dtype=torch.long, device=q.device)
        y = self._self_attention(block.self_attn, q=q, k=k_full, v=v_full, q_lens=q_lens, k_lens=k_lens)
        y = block.self_attn.o(y.flatten(2))

        with amp.autocast(device_type="cuda", dtype=torch.float32):
            x_active = x_active + y * e2
        x_active = x_active + block.cross_attn(block.norm3(x_active), context, context_lens)
        y_ffn = block.ffn(block.norm2(x_active).float() * (1 + e4) + e3)
        with amp.autocast(device_type="cuda", dtype=torch.float32):
            x_active = x_active + y_ffn * e5
        return x_active

    def _has_cache(self, module, branch_key: str) -> bool:
        if branch_key not in self.residual_cache:
            return False
        for layer_idx in range(len(module.blocks)):
            if (branch_key, layer_idx) not in self.k_cache or (branch_key, layer_idx) not in self.v_cache:
                return False
        return True

    @staticmethod
    def _active_positions_for_tokens(active_idx, token_indices):
        if active_idx.numel() == 0 or token_indices.numel() == 0:
            return torch.empty(0, device=active_idx.device, dtype=torch.long)
        if token_indices.numel() == active_idx.numel() and torch.equal(token_indices, active_idx):
            return torch.arange(active_idx.numel(), device=active_idx.device, dtype=torch.long)
        positions = torch.searchsorted(active_idx, token_indices)
        in_bounds = positions < active_idx.numel()
        if not bool(in_bounds.any()):
            return torch.empty(0, device=active_idx.device, dtype=torch.long)
        positions = positions[in_bounds]
        token_indices = token_indices[in_bounds]
        matches = active_idx.index_select(0, positions) == token_indices
        if not bool(matches.any()):
            return torch.empty(0, device=active_idx.device, dtype=torch.long)
        return positions[matches]

    @staticmethod
    def _rope_apply_indexed(x, token_indices, grid_size, freqs):
        _, _, n_heads, head_dim = x.shape
        half = head_dim // 2
        f, h, w = [int(v) for v in grid_size.tolist()]
        hw = h * w
        idx = token_indices.long()
        f_idx = torch.div(idx, hw, rounding_mode="floor")
        rem = idx - f_idx * hw
        h_idx = torch.div(rem, w, rounding_mode="floor")
        w_idx = rem - h_idx * w

        freqs_split = freqs.split([half - 2 * (half // 3), half // 3, half // 3], dim=1)
        f_part = freqs_split[0].to(x.device).index_select(0, f_idx)
        h_part = freqs_split[1].to(x.device).index_select(0, h_idx)
        w_part = freqs_split[2].to(x.device).index_select(0, w_idx)
        freq_token = torch.cat([f_part, h_part, w_part], dim=1).unsqueeze(1)

        x_complex = torch.view_as_complex(x[0].to(torch.float64).reshape(x.size(1), n_heads, -1, 2))
        out = torch.view_as_real(x_complex * freq_token).flatten(2).unsqueeze(0)
        return out.float()

    @staticmethod
    def _self_attention(self_attn, q, k, v, q_lens, k_lens):
        half_dtypes = (torch.float16, torch.bfloat16)

        def half(x):
            return x if x.dtype in half_dtypes else x.to(torch.bfloat16)

        attn_func = getattr(self_attn, "attn_func", None)
        if attn_func is None:
            return flash_attention(
                q=q,
                k=k,
                v=v,
                q_lens=q_lens,
                k_lens=k_lens,
                window_size=self_attn.window_size,
            )

        q_lens = q_lens.to(device=q.device, dtype=torch.int32)
        k_lens = k_lens.to(device=q.device, dtype=torch.int32)
        cu_seqlens_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32)
        cu_seqlens_k = torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32)
        out = attn_func(
            q=half(torch.cat([u[:v_] for u, v_ in zip(q, q_lens)])),
            k=half(torch.cat([u[:v_] for u, v_ in zip(k, k_lens)])),
            v=half(torch.cat([u[:v_] for u, v_ in zip(v, k_lens)])),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=int(q.size(1)),
            max_seqlen_k=int(k.size(1)),
            window_size=self_attn.window_size,
        )[0]
        return out.unflatten(0, (q.size(0), q.size(1))).type(q.dtype)
