import torch
import time
from typing import Optional, Tuple
from contextlib import nullcontext
from torch.nn.attention import SDPBackend, sdpa_kernel
from chitu_diffusion.core.distributed.parallel_state import get_cp_group, get_up_group

from logging import getLogger
from chitu_diffusion.runtime.parallel_utils import update_out_and_lse, squeeze_and_transpose
from chitu_diffusion.observability.timer import Timer

# try:
#     import flash_attn_interface       
#     FLASH_ATTN_3_AVAILABLE = True
# except ModuleNotFoundError:
FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn                   
    FLASH_ATTN_2_AVAILABLE = True
except ImportError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    import flashinfer
    FLASHINFER_AVAILABLE = True
except ImportError:
    FLASHINFER_AVAILABLE = False

try:
    import sageattention
    from sageattention import sageattn,sageattn_varlen
    SAGE_ATTENTION_AVAILABLE = True
except ImportError:
    SAGE_ATTENTION_AVAILABLE = False

try:
    import spas_sage_attn
    from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda
    SPAS_SAGE_ATTN_AVAILABLE = True
except ImportError:
    SPAS_SAGE_ATTN_AVAILABLE = False

logger = getLogger(__name__)
class DiffusionAttnBackend:
    """
    Unified backend for different attention implementations.
    Priority order:
    1. User specified attention type (if available)
    2. Fallback to PyTorch SDPA as default
    
    Supported types:
    - auto: prefer sparge, then sage, then flashinfer, then flash_attn, then torch_sdpa
    - flash_attn: Flash Attention v2
    - flashinfer: FlashInfer prefill attention
    - sage: SageAttention
    - sparge: Sparge/Sparse SageAttention
    - torch_sdpa: PyTorch scaled_dot_product_attention with PyTorch's kernel auto selection
    - torch_sdpa_math: PyTorch scaled_dot_product_attention forced to the math kernel
    """

    def __init__(self, attn_type: str = "torch_sdpa") -> None:
        self.impl = None
        self.topk = 0.5
        requested = self._normalize_attn_type(attn_type)

        if requested == "auto":
            for candidate in ("sparge", "sage", "flashinfer", "flash_attn", "torch_sdpa"):
                if self._is_available(candidate):
                    self.impl = candidate
                    break
        else:
            if not self._is_available(requested):
                raise RuntimeError(
                    f"Requested attention backend '{requested}' is not available. "
                    f"Availability: {self.availability_report()}."
                )
            self.impl = requested

        if self.impl is None:
            raise RuntimeError(f"No attention backend available. Availability: {self.availability_report()}.")

        logger.info("Using %s attention backend", self.impl)

    @staticmethod
    def _normalize_attn_type(attn_type: str) -> str:
        name = str(attn_type or "auto").strip().lower()
        aliases = {
            "flash": "flash_attn",
            "flash2": "flash_attn",
            "flash_v2": "flash_attn",
            "fa2": "flash_attn",
            "fi": "flashinfer",
            "flash_infer": "flashinfer",
            "sdpa": "torch_sdpa",
            "torch": "torch_sdpa",
            "ref": "torch_sdpa",
            "math": "torch_sdpa_math",
            "sdpa_math": "torch_sdpa_math",
            "torch_math": "torch_sdpa_math",
            "sparse": "sparge",
            "spas_sage": "sparge",
        }
        return aliases.get(name, name)

    @staticmethod
    def _is_available(backend: str) -> bool:
        if backend == "flash_attn":
            return FLASH_ATTN_2_AVAILABLE
        if backend == "flashinfer":
            return FLASHINFER_AVAILABLE
        if backend == "sage":
            return SAGE_ATTENTION_AVAILABLE
        if backend == "sparge":
            return SPAS_SAGE_ATTN_AVAILABLE
        if backend in {"torch_sdpa", "torch_sdpa_math"}:
            return hasattr(torch.nn.functional, "scaled_dot_product_attention")
        return False

    @staticmethod
    def availability_report() -> str:
        return (
            f"flash_attn={FLASH_ATTN_2_AVAILABLE}, "
            f"flashinfer={FLASHINFER_AVAILABLE}, "
            f"sage={SAGE_ATTENTION_AVAILABLE}, "
            f"sparge={SPAS_SAGE_ATTN_AVAILABLE}, "
            f"torch_sdpa={hasattr(torch.nn.functional, 'scaled_dot_product_attention')}"
        )

    # ------------- 统一入口 -------------
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
        dropout_p: float = 0.0,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[int, int] = (-1, -1),
        deterministic: bool = False,
        return_attn_probs: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns
        -------
        out : torch.Tensor
        softmax_lse : torch.Tensor, optional
        attn_probs : torch.Tensor, optional
        """
        use_varlen = cu_seqlens_q is not None

        if self.impl == "sparge":
            return self._fwd_sparge(
                q, k, v,
                cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k,
                dropout_p, softmax_scale,
                causal, window_size,
                deterministic, return_attn_probs,
                use_varlen,
            )
        elif self.impl == "sage":
            return self._fwd_sage(
                q, k, v,
                cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k,
                dropout_p, softmax_scale,
                causal, window_size,
                deterministic, return_attn_probs,
                use_varlen,
            )

        elif self.impl == "flash_attn":
            return self._fwd_v2(
                q, k, v,
                cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k,
                dropout_p, softmax_scale,
                causal, window_size,
                deterministic, return_attn_probs,
                use_varlen,
            )
        elif self.impl == "flashinfer":
            return self._fwd_flashinfer(
                q, k, v,
                cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k,
                dropout_p, softmax_scale,
                causal, window_size,
                return_attn_probs,
                use_varlen,
            )
        elif self.impl in {"torch_sdpa", "torch_sdpa_math"}:
            return self._fwd_torch_sdpa(
                q, k, v,
                cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k,
                dropout_p, softmax_scale,
                causal,
                use_varlen,
                return_attn_probs,
            )
        raise RuntimeError(f"Unknown attention backend implementation '{self.impl}'.")

    def _fwd_sparge(
        self,
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        dropout_p, softmax_scale,
        causal, window_size,
        deterministic,
        return_attn_probs,
        use_varlen: bool,
    ):
        if use_varlen:
            raise NotImplementedError("SPAS SAGE Attention does not support variable length sequences yet.")
        
        else:
            if return_attn_probs:
                logger.warning("[Not implemented] Sparge Attention varlen does not support 'return_attn_probs', which may cause error in context parallelism.")

            pvthreshd = torch.full((q.size(-2),), 50.0, dtype=torch.float32, device=q.device)
            out = spas_sage2_attn_meansim_topk_cuda(
                q,
                k,
                v,
                topk=self.topk,
                pvthreshd=pvthreshd,
                is_causal=causal,
                scale=softmax_scale,
                tensor_layout="NHD",
            )
       
            
        return out, None, None
    
    def _fwd_sage(
        self,
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        dropout_p, softmax_scale,
        causal, window_size,
        deterministic,
        return_attn_probs,
        use_varlen: bool,
    ):
        if use_varlen:
            if return_attn_probs:
                logger.warning("[Not implemented] Sage Attention varlen does not support 'return_attn_probs', which may cause error in context parallelism.")
            out = sageattn_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, is_causal=causal,tensor_layout="NHD")
            lse = None
        
        else:
            if return_attn_probs:
                out, lse = sageattn(q, k, v, is_causal=causal,tensor_layout="NHD", return_lse=return_attn_probs)
            else:
                out = sageattn(q, k, v, is_causal=causal,tensor_layout="NHD", return_lse=return_attn_probs)
                lse = None

       
            
        return out, lse, None

    def _fwd_torch_sdpa(
        self,
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        use_varlen: bool,
        return_attn_probs: bool = False,
    ):
        if use_varlen:
            raise NotImplementedError("torch_sdpa backend does not support varlen attention in ChituDiffusion.")
        q_in = q.transpose(1, 2)
        k_in = k.transpose(1, 2)
        v_in = v.transpose(1, 2)
        # Ring context parallelism needs the per-block logsumexp to merge partial
        # attention outputs across ring steps. The public SDPA API does not expose
        # the LSE, so route through the ATen flash op (Dynamo-traceable, also
        # CUDA-Graph capturable) which returns (out, logsumexp). The math fallback
        # cannot return LSE, so it is unsupported on the ring path.
        if return_attn_probs:
            if self.impl == "torch_sdpa_math":
                raise NotImplementedError(
                    "torch_sdpa_math cannot return logsumexp; use torch_sdpa for ring CP."
                )
            out, lse = torch.ops.aten._scaled_dot_product_flash_attention(
                q_in, k_in, v_in,
                dropout_p=dropout_p,
                is_causal=causal,
                return_debug_mask=False,
                scale=softmax_scale,
            )[:2]
            return out.transpose(1, 2).contiguous(), lse, None
        kernel_ctx = sdpa_kernel(SDPBackend.MATH) if self.impl == "torch_sdpa_math" else nullcontext()
        with kernel_ctx:
            out = torch.nn.functional.scaled_dot_product_attention(
                q_in,
                k_in,
                v_in,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=causal,
                scale=softmax_scale,
            )
        return out.transpose(1, 2).contiguous(), None, None

    def _fwd_flashinfer(
        self,
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        return_attn_probs,
        use_varlen: bool,
    ):
        if dropout_p != 0.0:
            raise NotImplementedError("FlashInfer backend does not support dropout in ChituDiffusion.")
        if window_size != (-1, -1):
            raise NotImplementedError("FlashInfer backend currently supports full attention only in ChituDiffusion.")

        if use_varlen:
            outputs = []
            lses = []
            batch = int(cu_seqlens_q.numel() - 1)
            for idx in range(batch):
                q_start = int(cu_seqlens_q[idx].item())
                q_end = int(cu_seqlens_q[idx + 1].item())
                k_start = int(cu_seqlens_k[idx].item())
                k_end = int(cu_seqlens_k[idx + 1].item())
                out, lse = self._flashinfer_single_prefill(
                    q[q_start:q_end],
                    k[k_start:k_end],
                    v[k_start:k_end],
                    softmax_scale=softmax_scale,
                    causal=causal,
                    return_lse=return_attn_probs,
                )
                outputs.append(out)
                if return_attn_probs:
                    lses.append(lse.transpose(0, 1).contiguous())
            out = torch.cat(outputs, dim=0) if outputs else q.new_empty(q.shape)
            lse = torch.cat(lses, dim=1) if return_attn_probs and lses else None
            return out, lse, None

        outputs = []
        lses = []
        for idx in range(q.shape[0]):
            out, lse = self._flashinfer_single_prefill(
                q[idx],
                k[idx],
                v[idx],
                softmax_scale=softmax_scale,
                causal=causal,
                return_lse=return_attn_probs,
            )
            outputs.append(out)
            if return_attn_probs:
                lses.append(lse.transpose(0, 1).contiguous())
        out = torch.stack(outputs, dim=0)
        lse = torch.stack(lses, dim=0) if return_attn_probs and lses else None
        return out, lse, None

    @staticmethod
    def _flashinfer_prefill_kwargs(softmax_scale, causal):
        kwargs = {
            "causal": causal,
            "kv_layout": "NHD",
        }
        if softmax_scale is not None:
            kwargs["sm_scale"] = softmax_scale
        return kwargs

    def _flashinfer_single_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        softmax_scale: Optional[float],
        causal: bool,
        return_lse: bool,
    ):
        kwargs = self._flashinfer_prefill_kwargs(softmax_scale, causal)
        if return_lse:
            out, lse = flashinfer.prefill.single_prefill_with_kv_cache_return_lse(q, k, v, **kwargs)
            return out, lse
        out = flashinfer.prefill.single_prefill_with_kv_cache(q, k, v, **kwargs)
        return out, None

    # ------------- v2 分支 -------------
    def _fwd_v2(
        self,
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        dropout_p, softmax_scale,
        causal, window_size,
        deterministic,
        return_attn_probs,
        use_varlen: bool,
    ):
        if use_varlen:
            fn = flash_attn.flash_attn_varlen_func
            kwargs = dict(
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic,
                return_attn_probs=return_attn_probs,
            )
            
        else:
            fn = flash_attn.flash_attn_func
            kwargs = dict(
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic,
                return_attn_probs=return_attn_probs,
            )
        if return_attn_probs:
            out, lse, s_mask = fn(q, k, v, **kwargs)
        else:
            out = fn(q, k, v, **kwargs)
            lse, s_mask = None, None
       
            
        return out, lse, s_mask

class DiffusionAttention_with_CP:
    def __init__(self, attn, ulysses_limit, ring_cudagraph: bool = False, cp_backend: str = "ucp"):
        self.attn = attn
        self.ulysses_limit = ulysses_limit
        self.group = get_cp_group()
        self.cp_size = self.group.group_size
        self.global_rank = torch.distributed.get_rank()
        self.local_chunk_id = self.group.rank_in_group
        # Optional: capture the whole context-parallel attention (ulysses all-to-all
        # + ring p2p loop + attn + lse merge) into a single CUDA Graph and replay it
        # per attention call, eliminating per-step NCCL launch / req.wait scheduling
        # overhead. Used for any CP path (pure ring, pure ulysses, unified SP);
        # non-varlen, cp_size>1.
        self.use_ring_cudagraph = bool(ring_cudagraph)
        self._ring_graph_cache: dict = {}
        self.cp_backend = self._validate_cp_backend(cp_backend)
        logger.info(
            "Context parallel attention backend: %s (cp_size=%s, up=%s)",
            self.cp_backend,
            self.cp_size,
            self.ulysses_limit,
        )

    @staticmethod
    def _validate_cp_backend(cp_backend: str) -> str:
        mode = str(cp_backend or "ucp").strip().lower()
        if mode not in {"agcp", "ucp"}:
            raise ValueError(f"Resolved CP backend must be one of agcp, ucp; got {cp_backend!r}.")
        return mode

    def uses_agcp(self) -> bool:
        return self.cp_backend == "agcp"

    def supports_fused_overlap(
        self,
        *,
        use_varlen: bool = False,
        return_attn_probs: bool = False,
        num_heads: Optional[int] = None,
    ) -> bool:
        """Whether the experimental async projection/collective overlap path can run."""
        ulysses_size = self.cp_size if self.cp_size <= self.ulysses_limit else self.ulysses_limit
        ring_steps = self.cp_size // ulysses_size
        return (
            self.cp_size > 1
            and self.cp_size % ulysses_size == 0
            and not self.uses_agcp()
            and not use_varlen
            and ring_steps == 1
            and ulysses_size > 1
            and not return_attn_probs
            and torch.cuda.is_available()
            and (num_heads is None or num_heads % ulysses_size == 0)
        )

    
    def async_ring_p2p_commit(self, tensors: Tuple[torch.Tensor, ...], src_rank: int, dst_rank: int):
        """Set up ring communication for sending and receiving tensors asynchronously.
    
        Args:
            tensors: Tuple of tensors to be sent
            dst_rank: Destination rank to send tensors to
            src_rank: Source rank to receive tensors from
            
        Returns:
            Tuple[torch.Tensor, ...]: Tuple of tensors to be received after wait
        """
        recv_tensors = []
        
        for tensor in tensors:
            send_tensor = tensor.contiguous()
            recv_size = send_tensor.shape
            recv_dtype = send_tensor.dtype
            self.group.p2p_isend(send_tensor, dst=dst_rank)
            next_tensor = self.group.p2p_irecv(size=recv_size, dtype=recv_dtype, src=src_rank)
            recv_tensors.append(next_tensor)
            
        self.group.p2p_commit()
        return tuple(recv_tensors)
    
    def async_ring_p2p_wait_and_update(self, recv_tensors: Tuple[torch.Tensor, ...]):
        """Wait for asynchronous communication to complete and return received tensors.
    
        Args:
            recv_tensors: Tuple of tensors returned from async_ring_p2p_commit
            
        Returns:
            Tuple[torch.Tensor, ...]: Tuple of received tensors after communication completes
        """
        self.group.p2p_wait()
        return recv_tensors

    def __call__(
        self,
        q: torch.Tensor, 
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
        dropout_p: float = 0,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[int, int] = (-1, -1),
        deterministic: bool = False,
        return_attn_probs: bool = False,
    ):
        '''
        Flash attention with context parallelism support.
        Input Q,K,V shape: 
        * flash attn func: (B, Seq_len, nheads, headdim)
        * varlen flash attn func: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        Will choose between flash_attn_func and flash_attn_varlen_func based on inputs
        '''
        cp_size = self.cp_size
        timer_start = None
        if Timer.is_enabled():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timer_start = time.perf_counter()

        ulysses_size = cp_size if cp_size <= self.ulysses_limit else self.ulysses_limit
        ring_steps = cp_size // ulysses_size
        if ring_steps > 1 and self.attn.impl in {"sage", "sparge"}:
            raise NotImplementedError(
                f"{self.attn.impl} context parallelism currently supports Ulysses only; "
                f"set infer.diffusion.up equal to cp_size ({cp_size})."
            )

        use_varlen = cu_seqlens_q is not None and cu_seqlens_k is not None and \
            max_seqlen_q is not None and max_seqlen_k is not None
        if self.uses_agcp():
            if use_varlen:
                raise NotImplementedError("AGCP does not support varlen attention yet.")
            k = self._all_gather_sequence(k, gather_dim=1)
            v = self._all_gather_sequence(v, gather_dim=1)
            out = self.attn(
                q,
                k,
                v,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic,
                return_attn_probs=False,
            )[0]
            return out, None, None

        attn_kwargs = dict(
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            use_varlen=use_varlen,
            ulysses_size=ulysses_size,
            ring_steps=ring_steps,
        )

        # CUDA Graph fast path: capture the whole context-parallel attention once
        # (ulysses all-to-all + ring p2p loop + all-to-all back) and replay it --
        # q/k/v shapes are identical across all layers and denoise steps. all-to-all
        # is a capturable NCCL collective just like the ring p2p, so this covers the
        # unified-SP path (ulysses_size > 1) as well as pure ring / pure ulysses.
        use_graph = (
            self.use_ring_cudagraph
            and cp_size > 1
            and not use_varlen
            and q.is_cuda
        )
        if use_graph:
            out = self._cp_forward_graphed(q, k, v, attn_kwargs)
        else:
            out = self._cp_forward(q, k, v, attn_kwargs)

        if timer_start is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - timer_start) * 1000.0
            Timer.record(f"cp{cp_size}_attn", elapsed_ms)

        return out, None, None

    @torch.compiler.disable
    def attention_fused(
        self,
        *,
        produce_q,
        produce_k,
        produce_v,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
        dropout_p: float = 0,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[int, int] = (-1, -1),
        deterministic: bool = False,
        return_attn_probs: bool = False,
        num_heads: Optional[int] = None,
    ):
        """Projection-aware CP attention for standard self-attention Q/K/V.

        The fast path is intentionally narrow: non-varlen pure Ulysses. Other
        modes materialize the producers and delegate to the regular CP path.
        """

        def serial():
            q = produce_q()
            k = produce_k()
            v = produce_v()
            return self(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic,
                return_attn_probs=return_attn_probs,
            )

        use_varlen = (
            cu_seqlens_q is not None
            or cu_seqlens_k is not None
            or max_seqlen_q is not None
            or max_seqlen_k is not None
        )
        if not self.supports_fused_overlap(
            use_varlen=use_varlen,
            return_attn_probs=return_attn_probs,
            num_heads=num_heads,
        ):
            return serial()

        ulysses_size = self.cp_size if self.cp_size <= self.ulysses_limit else self.ulysses_limit
        ulysses_group = get_up_group(ulysses_size)
        seq_dim, head_dim = 1, 2
        cur = torch.cuda.current_stream()
        cs = self._get_cp_comm_stream()

        def a2a_on_comm(t):
            cs.wait_stream(cur)
            t.record_stream(cs)
            with torch.cuda.stream(cs):
                out = ulysses_group.all_to_all(t, head_dim, seq_dim)
            ev = torch.cuda.Event()
            ev.record(cs)
            return out, ev

        k = produce_k()
        a2a_k, ev_k = a2a_on_comm(k)

        v = produce_v()
        a2a_v, ev_v = a2a_on_comm(v)

        q = produce_q()
        a2a_q, ev_q = a2a_on_comm(q)

        for ev in (ev_k, ev_v, ev_q):
            cur.wait_event(ev)
        for t in (a2a_q, a2a_k, a2a_v):
            t.record_stream(cur)

        out = self.attn(
            a2a_q,
            a2a_k,
            a2a_v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            return_attn_probs=False,
        )[0]
        out = ulysses_group.all_to_all(out, seq_dim, head_dim)
        return out, None, None

    def _all_gather_sequence(self, tensor: torch.Tensor, gather_dim: int) -> torch.Tensor:
        pieces = [torch.empty_like(tensor) for _ in range(self.cp_size)]
        torch.distributed.all_gather(pieces, tensor.contiguous(), group=self.group.gpu_group)
        return torch.cat(pieces, dim=gather_dim).contiguous()

    def _cp_forward(self, q, k, v, attn_kwargs: dict) -> torch.Tensor:
        """Core context-parallel attention (ulysses all-to-all + ring p2p loop).

        Returns the local attention output tensor. Side-effect free w.r.t. Python
        state except the comm group's transient p2p op/req buffers (cleared each step),
        which makes it safe to run under CUDA Graph capture.
        """
        cp_size = self.cp_size
        local_rank = self.local_chunk_id
        ulysses_size = attn_kwargs["ulysses_size"]
        ring_steps = attn_kwargs["ring_steps"]
        use_varlen = attn_kwargs["use_varlen"]
        cu_seqlens_q = attn_kwargs["cu_seqlens_q"]
        cu_seqlens_k = attn_kwargs["cu_seqlens_k"]
        max_seqlen_q = attn_kwargs["max_seqlen_q"]
        max_seqlen_k = attn_kwargs["max_seqlen_k"]

        if ulysses_size > 1:
            ulysses_group = get_up_group(ulysses_size)

        if use_varlen:
            seq_dim = 0
            head_dim = 1
            max_seqlen_q = max_seqlen_q * ulysses_size
            max_seqlen_k = max_seqlen_k * ulysses_size
        else:
            seq_dim = 1
            head_dim = 2

        ring_next_rank = (local_rank - ulysses_size) % cp_size
        ring_prev_rank = (local_rank + ulysses_size) % cp_size
        out, lse = None, None
        fresh_out, fresh_lse = None, None

        if ulysses_size > 1:
            q = ulysses_group.all_to_all(q, head_dim, seq_dim)
            k = ulysses_group.all_to_all(k, head_dim, seq_dim)
            v = ulysses_group.all_to_all(v, head_dim, seq_dim)

        for ring_step in range(ring_steps):
            if ring_step + 1 != ring_steps:
                if use_varlen:
                    data_pack = (k, v, cu_seqlens_k)
                else:
                    data_pack = (k, v)

                nxt_data_pack = self.async_ring_p2p_commit(
                    data_pack,
                    src_rank=ring_prev_rank,
                    dst_rank=ring_next_rank,
                )

            block_out, block_lse, _ = self.attn(
                q, k, v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p=attn_kwargs["dropout_p"],
                softmax_scale=attn_kwargs["softmax_scale"],
                causal=attn_kwargs["causal"],
                window_size=attn_kwargs["window_size"],
                deterministic=attn_kwargs["deterministic"],
                return_attn_probs=ring_steps > 1,
            )

            if ring_steps == 1 and block_lse is None:
                fresh_out = block_out
                fresh_lse = None
            else:
                fresh_out, fresh_lse = update_out_and_lse(fresh_out, fresh_lse, block_out, block_lse)

            if ring_step + 1 != ring_steps:
                if use_varlen:
                    k, v, cu_seqlens_k = self.async_ring_p2p_wait_and_update(nxt_data_pack)
                else:
                    k, v = self.async_ring_p2p_wait_and_update(nxt_data_pack)

            if fresh_lse is not None:
                fresh_lse = squeeze_and_transpose(fresh_lse)

            if ulysses_size > 1:
                fresh_out = ulysses_group.all_to_all(fresh_out, seq_dim, head_dim)
                if fresh_lse is not None:
                    fresh_lse = ulysses_group.all_to_all(fresh_lse, head_dim, seq_dim)

            if fresh_lse is None:
                out = fresh_out
                lse = None
            else:
                out, lse = update_out_and_lse(out, lse, fresh_out, fresh_lse)
            fresh_out, fresh_lse = None, None

        return out

    @torch.compiler.disable
    def cp_attn_with_full_txt(
        self,
        txt_q: torch.Tensor,
        txt_k: torch.Tensor,
        txt_v: torch.Tensor,
        img_q: torch.Tensor,
        img_k: torch.Tensor,
        img_v: torch.Tensor,
        *,
        image_seq_len: Optional[int] = None,
        dropout_p: float = 0.0,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[int, int] = (-1, -1),
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Qwen-Image CP attention with replicated text and sharded image tokens.

        Text Q/K/V stay full and replicated on every CP rank. Image Q/K/V are
        sharded by sequence. The first block attends to local text+image K/V;
        later ring steps stream only remote image K/V chunks. Outputs are
        returned as (full text output, local image output).
        """
        if causal:
            raise NotImplementedError("Qwen full-text CP attention only supports non-causal attention.")
        if self.cp_size <= 1:
            q = torch.cat([txt_q, img_q], dim=1)
            k = torch.cat([txt_k, img_k], dim=1)
            v = torch.cat([txt_v, img_v], dim=1)
            out = self.attn(
                q,
                k,
                v,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=False,
                window_size=window_size,
                deterministic=deterministic,
                return_attn_probs=False,
            )[0]
            return out[:, : txt_q.shape[1]], out[:, txt_q.shape[1] :]

        if self.uses_agcp():
            img_seq_len = int(image_seq_len if image_seq_len is not None else img_k.shape[1] * self.cp_size)
            full_img_key = self._all_gather_sequence(img_k, gather_dim=1)[:, :img_seq_len]
            full_img_value = self._all_gather_sequence(img_v, gather_dim=1)[:, :img_seq_len]
            q = torch.cat([txt_q, img_q], dim=1)
            k = torch.cat([txt_k, full_img_key], dim=1)
            v = torch.cat([txt_v, full_img_value], dim=1)
            out = self.attn(
                q,
                k,
                v,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=False,
                window_size=window_size,
                deterministic=deterministic,
                return_attn_probs=False,
            )[0]
            txt_len = txt_q.shape[1]
            return out[:, :txt_len], out[:, txt_len:]

        ulysses_size = self.cp_size if self.cp_size <= self.ulysses_limit else self.ulysses_limit
        ring_steps = self.cp_size // ulysses_size
        if ring_steps > 1 and self.attn.impl in {"sage", "sparge", "torch_sdpa_math"}:
            raise NotImplementedError(
                f"{self.attn.impl} cannot provide LSE for Qwen full-text ring attention."
            )

        local_rank = self.local_chunk_id
        ring_next_rank = (local_rank - ulysses_size) % self.cp_size
        ring_prev_rank = (local_rank + ulysses_size) % self.cp_size
        seq_dim = 1
        head_dim = 2
        if ulysses_size > 1:
            ulysses_group = get_up_group(ulysses_size)
            ulysses_rank = ulysses_group.rank_in_group
            txt_q = torch.tensor_split(txt_q, ulysses_size, head_dim)[ulysses_rank].contiguous()
            txt_k = torch.tensor_split(txt_k, ulysses_size, head_dim)[ulysses_rank].contiguous()
            txt_v = torch.tensor_split(txt_v, ulysses_size, head_dim)[ulysses_rank].contiguous()
            img_q = ulysses_group.all_to_all(img_q, head_dim, seq_dim)
            img_k = ulysses_group.all_to_all(img_k, head_dim, seq_dim)
            img_v = ulysses_group.all_to_all(img_v, head_dim, seq_dim)

        img_seq_len = int(image_seq_len if image_seq_len is not None else img_k.shape[1] * self.cp_size)
        original_local_img_len = img_q.shape[1] // ulysses_size if ulysses_size > 1 else img_q.shape[1]
        ring_block_seq = img_k.shape[1]
        local_img_len = img_q.shape[1]
        txt_len = txt_q.shape[1]
        q = torch.cat([txt_q, img_q], dim=1)

        out, lse = None, None
        cur_img_k, cur_img_v = img_k, img_v
        ring_block = local_rank // ulysses_size
        for ring_step in range(ring_steps):
            owner_block = (ring_block + ring_step) % ring_steps
            start = owner_block * ring_block_seq
            valid_len = max(0, min(cur_img_k.shape[1], img_seq_len - start))

            if ring_step + 1 != ring_steps:
                nxt_data_pack = self.async_ring_p2p_commit(
                    (cur_img_k, cur_img_v),
                    src_rank=ring_prev_rank,
                    dst_rank=ring_next_rank,
                )

            if ring_step == 0:
                block_k = torch.cat([txt_k, cur_img_k[:, :valid_len]], dim=1)
                block_v = torch.cat([txt_v, cur_img_v[:, :valid_len]], dim=1)
            else:
                block_k = cur_img_k[:, :valid_len]
                block_v = cur_img_v[:, :valid_len]

            if block_k.shape[1] > 0:
                block_out, block_lse, _ = self.attn(
                    q,
                    block_k,
                    block_v,
                    dropout_p=dropout_p,
                    softmax_scale=softmax_scale,
                    causal=False,
                    window_size=window_size,
                    deterministic=deterministic,
                    return_attn_probs=ring_steps > 1,
                )
                if ring_steps == 1 and block_lse is None:
                    out, lse = block_out, None
                else:
                    out, lse = update_out_and_lse(out, lse, block_out, block_lse)

            if ring_step + 1 != ring_steps:
                cur_img_k, cur_img_v = self.async_ring_p2p_wait_and_update(nxt_data_pack)

        if out is None:
            raise RuntimeError("Qwen full-text CP attention produced no output blocks.")
        txt_out = out[:, :txt_len]
        img_out = out[:, txt_len : txt_len + local_img_len]
        if ulysses_size > 1:
            img_out = ulysses_group.all_to_all(img_out, seq_dim, head_dim)
            img_out = img_out[:, :original_local_img_len]
            gathered_txt = [torch.empty_like(txt_out) for _ in range(ulysses_size)]
            torch.distributed.all_gather(gathered_txt, txt_out.contiguous(), group=ulysses_group.gpu_group)
            txt_out = torch.cat(gathered_txt, dim=head_dim).contiguous()
        return txt_out, img_out

    @torch.compiler.disable
    def cp_attn_with_full_txt_fused(
        self,
        *,
        produce_txt_q,
        produce_txt_k,
        produce_txt_v,
        produce_img_q,
        produce_img_k,
        produce_img_v,
        image_seq_len: Optional[int] = None,
        dropout_p: float = 0.0,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[int, int] = (-1, -1),
        deterministic: bool = False,
        num_heads: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Projection-aware full-text CP API surface.

        Producers are owned by model processors (projection/norm/RoPE). This
        method owns the CP schedule. First async implementation targets pure
        Ulysses full-text attention, matching the harness-proven schedule; other
        modes/features conservatively materialize and delegate.
        """
        if causal:
            raise NotImplementedError("Qwen full-text CP attention only supports non-causal attention.")

        def serial():
            txt_q = produce_txt_q()
            txt_k = produce_txt_k()
            txt_v = produce_txt_v()
            img_q = produce_img_q()
            img_k = produce_img_k()
            img_v = produce_img_v()
            return self.cp_attn_with_full_txt(
                txt_q,
                txt_k,
                txt_v,
                img_q,
                img_k,
                img_v,
                image_seq_len=image_seq_len,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=False,
                window_size=window_size,
                deterministic=deterministic,
            )

        if self.cp_size <= 1 or self.uses_agcp():
            return serial()

        ulysses_size = self.cp_size if self.cp_size <= self.ulysses_limit else self.ulysses_limit
        if num_heads is not None and ulysses_size > 1 and num_heads % ulysses_size != 0:
            return serial()
        if self.supports_fused_overlap(num_heads=num_heads):
            return self._cp_full_txt_ulysses_fused(
                produce_txt_q=produce_txt_q,
                produce_txt_k=produce_txt_k,
                produce_txt_v=produce_txt_v,
                produce_img_q=produce_img_q,
                produce_img_k=produce_img_k,
                produce_img_v=produce_img_v,
                image_seq_len=image_seq_len,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                window_size=window_size,
                deterministic=deterministic,
                ulysses_size=ulysses_size,
            )

        return serial()

    def _get_cp_comm_stream(self):
        stream = getattr(self, "_cp_comm_stream", None)
        if stream is None:
            stream = torch.cuda.Stream()
            self._cp_comm_stream = stream
        return stream

    def _cp_full_txt_ulysses_fused(
        self,
        *,
        produce_txt_q,
        produce_txt_k,
        produce_txt_v,
        produce_img_q,
        produce_img_k,
        produce_img_v,
        image_seq_len: Optional[int],
        dropout_p: float,
        softmax_scale: Optional[float],
        window_size: Tuple[int, int],
        deterministic: bool,
        ulysses_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ulysses_group = get_up_group(ulysses_size)
        ulysses_rank = ulysses_group.rank_in_group
        seq_dim, head_dim = 1, 2
        cur = torch.cuda.current_stream()
        cs = self._get_cp_comm_stream()

        def a2a_on_comm(t):
            if t.shape[2] % ulysses_size != 0:
                raise ValueError(
                    f"Ulysses CP needs heads ({t.shape[2]}) divisible by up size ({ulysses_size})"
                )
            cs.wait_stream(cur)
            t.record_stream(cs)
            with torch.cuda.stream(cs):
                out = ulysses_group.all_to_all(t, head_dim, seq_dim)
            ev = torch.cuda.Event()
            ev.record(cs)
            return out, ev

        img_k = produce_img_k()
        a2a_k, ev_k = a2a_on_comm(img_k)

        img_v = produce_img_v()
        a2a_v, ev_v = a2a_on_comm(img_v)

        img_q = produce_img_q()
        a2a_q, ev_q = a2a_on_comm(img_q)

        txt_q = produce_txt_q()
        txt_k = produce_txt_k()
        txt_v = produce_txt_v()

        for ev in (ev_k, ev_v, ev_q):
            cur.wait_event(ev)
        for t in (a2a_k, a2a_v, a2a_q):
            t.record_stream(cur)

        txt_q = torch.tensor_split(txt_q, ulysses_size, head_dim)[ulysses_rank].contiguous()
        txt_k = torch.tensor_split(txt_k, ulysses_size, head_dim)[ulysses_rank].contiguous()
        txt_v = torch.tensor_split(txt_v, ulysses_size, head_dim)[ulysses_rank].contiguous()

        img_seq_len = int(image_seq_len if image_seq_len is not None else a2a_k.shape[1])
        original_local_img_len = a2a_q.shape[1] // ulysses_size
        full_img_len = a2a_q.shape[1]
        valid_kv_len = max(0, min(a2a_k.shape[1], img_seq_len))
        txt_len = txt_q.shape[1]
        q = torch.cat([txt_q, a2a_q], dim=1)
        k = torch.cat([txt_k, a2a_k[:, :valid_kv_len]], dim=1)
        v = torch.cat([txt_v, a2a_v[:, :valid_kv_len]], dim=1)

        out = self.attn(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=False,
            window_size=window_size,
            deterministic=deterministic,
            return_attn_probs=False,
        )[0]

        txt_out = out[:, :txt_len]
        img_out = out[:, txt_len:txt_len + full_img_len]
        img_out = ulysses_group.all_to_all(img_out, seq_dim, head_dim)
        img_out = img_out[:, :original_local_img_len]
        gathered_txt = [torch.empty_like(txt_out) for _ in range(ulysses_size)]
        torch.distributed.all_gather(gathered_txt, txt_out.contiguous(), group=ulysses_group.gpu_group)
        txt_out = torch.cat(gathered_txt, dim=head_dim).contiguous()
        return txt_out, img_out

    def _cp_forward_graphed(self, q, k, v, attn_kwargs: dict) -> torch.Tensor:
        """Capture the whole context-parallel attention into a CUDA Graph and replay it.

        The captured graph is keyed by tensor shape/dtype, so a single graph is
        reused across all attention layers and denoise steps (identical shapes).
        Inputs are copied into static buffers before replay; the static output is
        cloned out so the next replay can safely overwrite it.
        """
        key = (tuple(q.shape), tuple(k.shape), tuple(v.shape), q.dtype,
               bool(attn_kwargs["causal"]), attn_kwargs["softmax_scale"])
        entry = self._ring_graph_cache.get(key)
        if entry is None:
            static_q = q.clone()
            static_k = k.clone()
            static_v = v.clone()

            # Warmup on a side stream (required before capture; also initializes
            # NCCL p2p connections so they are not established during capture).
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._cp_forward(static_q, static_k, static_v, attn_kwargs)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_out = self._cp_forward(static_q, static_k, static_v, attn_kwargs)
            torch.cuda.synchronize()

            entry = {
                "graph": graph,
                "q": static_q,
                "k": static_k,
                "v": static_v,
                "out": static_out,
            }
            self._ring_graph_cache[key] = entry
            logger.info(
                "[ring-cudagraph] captured ring loop graph for shape q=%s (cp_size=%d)",
                tuple(q.shape),
                self.cp_size,
            )

        entry["q"].copy_(q)
        entry["k"].copy_(k)
        entry["v"].copy_(v)
        entry["graph"].replay()
        return entry["out"].clone()
            
