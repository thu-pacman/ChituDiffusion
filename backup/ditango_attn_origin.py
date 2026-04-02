import torch
from typing import Optional, Tuple
from logging import getLogger

from chitu_core.distributed.parallel_state import get_cp_group, get_up_group
from chitu_diffusion.backend import DiffusionBackend
from chitu_diffusion.modules.attention.diffusion_attn_backend import DiffusionAttnBackend
from chitu_diffusion.utils.shared_utils import update_out_and_lse, squeeze_and_transpose, async_ring_p2p_commit, async_ring_p2p_wait_and_update


logger = getLogger(__name__)


def get_timestep():
    return DiffusionBackend.generator.current_task.buffer.current_step


class AttnState_Cache:
    def __init__(self, cp_size, layer_id):
        self.cp_size = cp_size
        self.curr_cp_stride = cp_size
        self.curr_block_num = cp_size
        self.layer_id = layer_id
        self.alpha = 0
        self.out_block_cache = [None] * self.curr_block_num
        self.lse_block_cache = [None] * self.curr_block_num
        self.block_age = [0] * self.curr_block_num
    
    def _merge_blocks(self, new_block_num: int):
        # logger.debug(f"Merging blocks from {self.curr_block_num} to {new_block_num}")
        assert self.curr_block_num % new_block_num == 0, f"Invalid block number {new_block_num} for merging."
        chunks_to_merge = self.curr_block_num // new_block_num
        new_out_block_cache, new_lse_block_cache = [None] * new_block_num, [None] * new_block_num
        new_block_age = [None] * new_block_num
        for i in range(new_block_num):
            merged_block_out, merged_block_lse = None, None
            merged_age = 0
            for j in range(chunks_to_merge):
                block_id = i * chunks_to_merge + j
                cached_out = self.out_block_cache[block_id]
                cached_lse = self.lse_block_cache[block_id]
                merged_age += self.block_age[block_id]
                if cached_lse is not None:
                    merged_block_out, merged_block_lse = update_out_and_lse(merged_block_out, merged_block_lse, cached_out, cached_lse)
            if merged_block_lse is not None:
                merged_block_lse = squeeze_and_transpose(merged_block_lse)
            new_out_block_cache[i] = merged_block_out
            new_lse_block_cache[i] = merged_block_lse
            new_block_age[i] = merged_age / chunks_to_merge
        self.out_block_cache = new_out_block_cache
        self.lse_block_cache = new_lse_block_cache
        self.block_age = new_block_age
    
    def adjust_cache_shape(self, new_cp_stride: int, num_chunks_per_block: int):
        assert self.cp_size % new_cp_stride == 0, "Unsupport cp stride size!"
        # 小到满，相等的情况
        if new_cp_stride == self.cp_size or new_cp_stride == self.curr_cp_stride: # 满，不存，在evict时会调整 / 相同步长，保持不变即可
            pass
        else:
            new_block_num = self.cp_size // num_chunks_per_block
            if self.curr_cp_stride == self.cp_size: # 满到小的情况
                self.curr_block_num = new_block_num
                self.out_block_cache = [None] * self.curr_block_num
                self.lse_block_cache = [None] * self.curr_block_num
                self.block_age = [0] * self.curr_block_num
                
            # elif new_block_num < self.curr_block_num: # 小到大的情况，需要merge，这种情况等cache更新以后处理
            #     self._merge_blocks(new_block_num)
            #     self.curr_block_num = new_block_num
            # self.curr_cp_stride = new_cp_stride
                 
    def merge_and_evict_blocks(self, new_cp_stride: int, target_block_id: int):
        # merge
        if self.curr_cp_stride == self.cp_size or new_cp_stride > self.curr_cp_stride: # 满到小 或者 小到大，合并
            new_block_num = self.cp_size // new_cp_stride
            self._merge_blocks(new_block_num)
            self.curr_block_num = new_block_num
            self.curr_cp_stride = new_cp_stride
        
        # evict
        if new_cp_stride == self.cp_size: # 小到满，全清
            self.out_block_cache = [None] * self.curr_block_num
            self.lse_block_cache = [None] * self.curr_block_num
            self.block_age = [0] * self.curr_block_num
        else:
            self.out_block_cache[target_block_id] = None
            self.lse_block_cache[target_block_id] = None
            
    def update_block_age(self):
        for id, block in enumerate(self.lse_block_cache):
            if block is not None:
                self.block_age[id] += 1 
        
    def get_block(self, block_id: int):
        if block_id >= len(self.out_block_cache):
            logger.error(f"{get_timestep()} | Cache is not ready but trying to get cached v.")
        cached_out = self.out_block_cache[block_id] 
        cached_lse = self.lse_block_cache[block_id]
        return cached_out, cached_lse
    
    def store_block(self, block_id, out, lse):
        self.out_block_cache[block_id] = out
        self.lse_block_cache[block_id] = lse
        self.block_age[block_id] = 1
            
    def clear(self):
        logger.warning("============== Clear Attention State Cache =================")
    
    def set_block_size_mb(self, tensor):
        original_shape = list(tensor.shape)
        
        new_shape = original_shape.copy()
        new_shape[-1] += 1
        
        expanded_tensor = torch.zeros(new_shape, dtype=torch.float32, device=tensor.device)
        
        if self.block_size_mb is None:
            block_size_mb = expanded_tensor.element_size() * expanded_tensor.nelement() / (1024 ** 2)
            self.block_size_mb = block_size_mb
        # logger.debug(f"{expanded_tensor.shape=} | Block size MB: {self.block_size_mb}")
        return self.block_size_mb
             
    def report_cache_status(self, layer_id: int):
        """
        Report the current status of the cache including:
        - Cache configuration
        - Block storage status
        - Memory usage information
        - Shape information of cached blocks
        """
        # 获取当前时间步
        timestep = get_timestep()
        
        # 计算已缓存的块数
        cached_blocks = sum(1 for out in self.out_block_cache if out is not None)
        
        # 计算单个块大小并找到第一个非空块作为示例
        block_size_mb = 0
        example_out = None
        example_lse = None
        
        for block_id in range(len(self.out_block_cache)):
            out = self.out_block_cache[block_id]
            lse = self.lse_block_cache[block_id]
            if out is not None:
                example_out = out
                example_lse = lse
                
                block_size_bytes = out.element_size() * out.nelement()
                if lse is not None:
                    block_size_bytes += lse.element_size() * lse.nelement()
                block_size_mb = block_size_bytes / (1024 * 1024)
                if self.block_size_mb != block_size_mb:
                    self.block_size_mb = block_size_mb
                break
        
        # 计算总缓存大小
        total_cache_size_mb = block_size_mb * cached_blocks
        
        # 获取当前GPU内存使用情况
        current_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        max_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
        total_memory = self.memory_constraint
        
        # 输出报告标题
        logger.info(f"===== Cache Status: Layer {layer_id}, Timestep {timestep} =====")
        
        # 基本配置信息
        logger.info(f"Maximum cp Size: {self.cp_size}")
        logger.info(f"Current cp Stride: {self.curr_cp_stride}")
        logger.info(f"Current Block Count: {self.curr_block_num}")
        
        # 缓存利用率信息
        logger.info(f"Filled Blocks: {cached_blocks}/{self.curr_block_num} ({cached_blocks/self.curr_block_num*100:.1f}% full)")
        
        # 生成块状态可视化 (O:有数据 X:空)
        block_status = ['O' if out is not None else 'X' for out in self.out_block_cache]
        block_status_str = ' '.join(block_status)
        logger.info(f"Block Status: [{block_status_str}]")
        
        # 内存使用信息
        logger.info(f"Block size: {block_size_mb:.2f} MB")
        logger.info(f"Total cache size: {total_cache_size_mb:.2f} MB")
        logger.info(f"GPU memory: {current_memory:.2f}/{total_memory:.2f} MB ({current_memory/total_memory*100:.1f}%)")
        
        if total_cache_size_mb > 0:
            logger.info(f"Cache percentage: {total_cache_size_mb/current_memory*100:.1f}% of GPU usage")
        
        # 张量形状信息 (直接报告，不需要额外参数)
        if cached_blocks > 0 and example_out is not None:
                
            logger.info("Block Tensor Shapes:")
            logger.info(f"  OUT tensor: {tuple(example_out.shape)}")
            if example_lse is not None:
                logger.info(f"  LSE tensor: {tuple(example_lse.shape)}")
            else:
                logger.info("  LSE tensor: None")
        
        logger.info("================================================")


class DitangoAttention:
    def __init__(self, ulysses_limit: int, layer_id: int):
        self.attn = DiffusionAttnBackend()
        self.group = get_cp_group()
        self.cp_size = self.group.group_size
        self.global_rank = torch.distributed.get_rank()
        self.local_chunk_id = self.group.rank_in_group
        self.layer_id = layer_id
        self.target_chunk_id = self.local_chunk_id
        self.cache = AttnState_Cache(self.cp_size, layer_id)
        self.reuse_phase_done = True   # 标记本轮的复用是否已经完成
        
        self.ulysses_limit = ulysses_limit
        self.measure_memory = False
        if self.global_rank == 0:
            logger.info(f"L{layer_id} | Using Ditango Attn.")
                  
    def print_layer0(self, message: str):
        enable = self.global_rank in [0]
        # enable = True
        if enable and self.layer_id == 0:
            logger.info(message)
    
            
    def get_ring_and_ulysses_steps(self, curr_cp_stride, next_cp_stride):
        ring_steps, ulysses_size, ring_steps_to_update = None, None, None
                   
        if curr_cp_stride == self.cp_size: # Need to cache more than one blocks
            ulysses_size = min(next_cp_stride, self.ulysses_limit)
            ring_steps = curr_cp_stride // ulysses_size
            ring_steps_to_update = 1 # 每次都要更新
        else: # Only need to cache 1 block
            if curr_cp_stride <= self.ulysses_limit: # 全面ulysses
                ulysses_size = curr_cp_stride
                ring_steps = 1
            else: # 机内尽量ulysses
                ulysses_size = curr_cp_stride & -curr_cp_stride  # 找到2的幂次因子
                ulysses_size = min(self.ulysses_limit, ulysses_size)  # 限制ulysses最大值
                ring_steps = curr_cp_stride // ulysses_size
            ring_steps_to_update = ring_steps # 更新一次
            
        return ring_steps, ulysses_size, ring_steps_to_update
    
    def get_block_info(self):
        curr_block_num = self.cache.curr_block_num
        local_block_id = self.local_chunk_id // (self.cp_size // curr_block_num)
        target_block_id = self.target_chunk_id // (self.cp_size // curr_block_num)
        return curr_block_num, local_block_id, target_block_id
        
    def reuse_cached_chunks(self, total_block_num, target_block_id):
        out, lse = None, None
        if self.reuse_phase_done:
            return out, lse
        for i in range(total_block_num - 1):
            cached_block_id = (target_block_id + 1 + i) % total_block_num
            cached_out, cached_lse = self.cache.get_block(cached_block_id)
            if cached_lse is not None:
                self.print_layer0(f"R{self.global_rank}T{get_timestep()}L{self.layer_id}| Overlapped get cached block {cached_block_id}.")
                out, lse = update_out_and_lse(out, lse, cached_out, cached_lse)
            else:
                assert total_block_num == self.cp_size and cached_block_id == self.local_chunk_id, f"R{self.global_rank}T{get_timestep()}L{self.layer_id} | Failed to get cached block {cached_block_id}, got NONE."
        self.reuse_phase_done = True
        return out, lse
    
    def _get_current_and_next_stride(self) -> Tuple:
        """
        FIXME
        Naive strategy: fix ratio except first step.
        """
        curr_cp_stride = self.cp_size // 2 if get_timestep() > 0 else self.cp_size
        next_cp_stride = self.cp_size // 2
        return curr_cp_stride, next_cp_stride
        
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
        # Note: input size should be (b, s, a, d)
        # =================== 0. Initialize parameters =================
        timestep = get_timestep()
        use_varlen = cu_seqlens_q is not None and cu_seqlens_k is not None and \
        max_seqlen_q is not None and max_seqlen_k is not None

        if use_varlen:
            seq_dim = 0
            head_dim = 1
            max_seqlen_q *= ulysses_size
            max_seqlen_k *= ulysses_size
            data_pack = (k, v, cu_seqlens_k)
        else:
            seq_dim = 1
            head_dim=2
            data_pack = (k, v)

        curr_cp_stride, next_cp_stride = self._get_current_and_next_stride()
       

        self.reuse_phase_done = False                
        assert self.cp_size % curr_cp_stride == 0, f"Does not support this cp stride {curr_cp_stride} for SP size {self.cp_size}"

        if curr_cp_stride == self.cp_size:
            self.reuse_phase_done = True
            self.target_chunk_id = self.local_chunk_id
            
        curr_block_num, local_block_id, target_block_id = self.get_block_info()
                    
        # ================== 2. Reuse stage: KV Layout Transform + local attention ======================= 
        local_block_id = self.local_chunk_id // curr_cp_stride # 当前进程属于哪个block
        target_block_id = self.target_chunk_id // curr_cp_stride # 表示本轮需要计算的block id
        
        if self.target_chunk_id != self.local_chunk_id: # need to get first kv
            block_id_stride = (target_block_id - local_block_id) % curr_block_num
            send_block_id = (local_block_id - block_id_stride) % curr_block_num
            inner_block_rank = self.local_chunk_id % curr_cp_stride
            
            recv_rank = target_block_id * curr_cp_stride + inner_block_rank
            send_rank = send_block_id * curr_cp_stride + inner_block_rank

            kv_transform_data_pack = async_ring_p2p_commit(
                self.group,
                data_pack,
                src_rank=recv_rank,
                dst_rank=send_rank
            )
            self.print_layer0(f"R{self.global_rank}T{timestep}L{self.layer_id}| Recving target chunk {self.target_chunk_id} from rank {recv_rank}.")
            
            # ============ 2.5 local attention overlap ==============
            
            out, lse = self.reuse_cached_chunks(
                total_block_num=curr_block_num,
                target_block_id=target_block_id)
                        
            block_out, block_lse, _  = self.attn(
                    q, k, v,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    dropout_p=dropout_p,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=window_size,
                    deterministic=deterministic,
                    return_attn_probs=True,
                )
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)
            if use_varlen:
                k, v, cu_seqlens_k = async_ring_p2p_wait_and_update(self.group, kv_transform_data_pack)
            else:        
                k, v = async_ring_p2p_wait_and_update(self.group, kv_transform_data_pack)
            data_pack = kv_transform_data_pack
        else:
            out, lse = self.reuse_cached_chunks(
                total_block_num=curr_block_num,
                target_block_id=target_block_id)
        
        # =================。 adjust cache and target block ================
        ring_steps, ulysses_size, ring_steps_to_update = self.get_ring_and_ulysses_steps(curr_cp_stride, next_cp_stride)
        self.cache.adjust_cache_shape(new_cp_stride=next_cp_stride, num_chunks_per_block=ulysses_size)
        curr_block_num, local_block_id, target_block_id = self.get_block_info()
            
        # ================= 3. Compute Stage: Grouped context parallelism ====================
        
        # logger.debug(f"R{self.global_rank}T{timestep}L{self.layer_id} | {curr_cp_stride=}, {next_cp_stride=}, {ring_steps=}, {ulysses_size=}, {ring_steps_to_update=},{query.shape=}")
        ring_offset = self.local_chunk_id // curr_cp_stride * curr_cp_stride
        ring_next_rank = (self.local_chunk_id - ulysses_size) % curr_cp_stride + ring_offset
        ring_prev_rank = (self.local_chunk_id + ulysses_size) % curr_cp_stride + ring_offset
        # logger.debug(f"R{self.global_rank}T{timestep}L{self.layer_id} | {ring_next_rank=}, {ring_prev_rank=}")
        fresh_out, fresh_lse = None, None
        ulysses_group = get_up_group(ulysses_size)
        self.print_layer0(f"Init | {q.shape=} {k.shape=} {v.shape=}")

        if ulysses_size > 1:
            q = ulysses_group.all_to_all(q, head_dim, seq_dim)  
            k = ulysses_group.all_to_all(k, head_dim, seq_dim)
            v = ulysses_group.all_to_all(v, head_dim, seq_dim)
        
        for ring_step in range(ring_steps):
            if use_varlen:
                data_pack = (k, v, cu_seqlens_k)
            else:
                data_pack = (k, v)

            if ring_step + 1 != ring_steps:
                nxt_data_pack = async_ring_p2p_commit(
                    self.group,
                    data_pack,
                    src_rank=ring_prev_rank,
                    dst_rank=ring_next_rank,
                )

            self.print_layer0(f"{ring_step=} | {q.shape=} {k.shape=} {v.shape=}")
            block_out, block_lse, _  = self.attn(
                    q, k, v,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    dropout_p=dropout_p,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=window_size,
                    deterministic=deterministic,
                    return_attn_probs=True,
                ) # out:(b,s,h,d) lse:(b,d,s)
            fresh_out, fresh_lse = update_out_and_lse(fresh_out, fresh_lse, block_out, block_lse)
            
            if ring_step + 1 != ring_steps:
                if use_varlen:
                    k, v, cu_seqlens_k = async_ring_p2p_wait_and_update(self.group, nxt_data_pack)
                else:
                    k, v = async_ring_p2p_wait_and_update(self.group, nxt_data_pack)
            
            if (ring_step + 1) % ring_steps_to_update == 0:  # 已经到了更新的时候，按照block size决定
                fresh_lse = squeeze_and_transpose(fresh_lse)
                if ulysses_size > 1:
                    fresh_out = ulysses_group.all_to_all(fresh_out, seq_dim, head_dim)
                    fresh_lse = ulysses_group.all_to_all(fresh_lse, head_dim, seq_dim)
                out, lse = update_out_and_lse(out, lse , fresh_out, fresh_lse) # 先更新到本层输出
                
                if next_cp_stride == self.cp_size or (curr_block_num == self.cp_size and target_block_id == self.local_chunk_id):
                    should_cache = False
                else:
                    should_cache = True # 什么情况下要储存？
                    
                if should_cache: # 再存起来
                    try:
                        self.print_layer0(f"R{self.global_rank}T{timestep}L{self.layer_id} | trying to save block {target_block_id}, {fresh_out.shape=}, {fresh_lse.shape=}")
                        self.cache.store_block(target_block_id, fresh_out, fresh_lse)
                        # cached_chunks.append(calc_chunks_per_block)
                    except IndexError:
                        logger.error(f"R{self.global_rank}T{timestep}L{self.layer_id} | Failed to save block {target_block_id}, {fresh_out.shape=}, {fresh_lse.shape=}, but {self.cache.curr_block_num=}")
                        exit()
                        
                target_block_id = (target_block_id + 1) % curr_block_num 
                # logger.debug(f"R{self.global_rank}T{timestep}L{self.layer_id} | Updated targetblock {target_block_id=}")
                
                fresh_out, fresh_lse = None, None # 只有在存下来后，才可以刷新fresh out lse
                # calc_chunks_per_block = []
                
        # another update strategy
        if next_cp_stride > curr_cp_stride: # merge情况，保留在当前block中
            stride_offset = self.local_chunk_id % next_cp_stride
            self.target_chunk_id = self.target_chunk_id  // next_cp_stride * next_cp_stride + stride_offset  
        else: # 相等，正常行进
            self.target_chunk_id = (self.target_chunk_id + curr_cp_stride) % self.cp_size
          
        if self.cache is not None:
            target_block_id = self.target_chunk_id // next_cp_stride
            self.cache.merge_and_evict_blocks(next_cp_stride, target_block_id)
            self.cache.update_block_age()

        return out, None, None 
            



