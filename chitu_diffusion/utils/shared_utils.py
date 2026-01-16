import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from chitu_core.distributed.comm_group import CommGroup

class SequencePadder:
    _padding_info: Dict = {}
    DEFAULT_NAME = "_default_"
    
    @staticmethod 
    def split_sequence_padding(tensor: torch.Tensor,
                              split_num: int,
                              split_dim: int = 0,
                              name: Optional[str] = None) -> List[torch.Tensor]:
        """Split tensor into split_num parts along specified dimension with padding if needed"""
        if name is None:
            name = SequencePadder.DEFAULT_NAME
            
        size = tensor.size(split_dim)
        split_size = (size + split_num - 1) // split_num  # 向上取整确保能分成split_num份
        
        if size % split_num != 0:
            pad_size = split_size * split_num - size
            pad_shape = list(tensor.shape)
            pad_shape[split_dim] = pad_size
            padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
            tensor = torch.cat([tensor, padding], dim=split_dim)
            SequencePadder._padding_info[name] = {'original_size': size, 'pad_size': pad_size}
    
        splits = torch.split(tensor, split_size, dim=split_dim)
        return list(splits)

    @staticmethod
    def remove_sequence_padding_and_concat(tensor_list: List[torch.Tensor],
                              gather_dim: int = 0,
                              name: Optional[str] = None) -> torch.Tensor:
        """Remove padding from split tensors based on padding info"""
        if name is None:
            # 如果没有提供name,使用_padding_info中的第一个key
            if len(SequencePadder._padding_info) > 0:
                name = next(iter(SequencePadder._padding_info))
            else:
                return torch.cat(tensor_list, dim=gather_dim)
                
        if name not in SequencePadder._padding_info:
            return torch.cat(tensor_list, dim=gather_dim)
            
        info = SequencePadder._padding_info[name]
        original_size = info['original_size']
        
        tensor = torch.cat(tensor_list, dim=gather_dim)
        slicing = [slice(None)] * tensor.dim()
        slicing[gather_dim] = slice(0, original_size)
        tensor = tensor[slicing]
        
        return tensor
    
@torch.jit.script
def transpose_and_unsqueeze(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(-2, -1).unsqueeze(-1)

@torch.jit.script
def squeeze_and_transpose(x: torch.Tensor) -> torch.Tensor:
    return x.squeeze(-1).transpose(-1, -2)

@torch.jit.script
def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    # For additional context and discussion, please refer to:
    # https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795

    block_out = block_out.to(torch.float32)
    block_lse = transpose_and_unsqueeze(block_lse)
    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)
    return out, lse


def update_out_and_lse(
    out: Optional[torch.Tensor], # （b, s, n, d）
    lse: Optional[torch.Tensor], # (b, s, n, 1)
    block_out: torch.Tensor, # (b, s, n, d)
    block_lse: torch.Tensor, # (b, n, s)
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    # output: out (b, s, n, d) lse (b, s, n, 1)        
    if out is None:
      out = block_out.to(torch.float32)
      lse = transpose_and_unsqueeze(block_lse)
    else:
      out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
      
    return out, lse


def async_ring_p2p_commit(group: CommGroup, tensors: Tuple[torch.Tensor, ...], src_rank: int, dst_rank: int):
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
        send_tensor = tensor
        recv_size = send_tensor.shape
        recv_dtype = send_tensor.dtype
        group.p2p_isend(send_tensor, dst=dst_rank)
        next_tensor = group.p2p_irecv(size=recv_size, dtype=recv_dtype, src=src_rank)
        recv_tensors.append(next_tensor)
        
    group.p2p_commit()
    return tuple(recv_tensors)

def async_ring_p2p_wait_and_update(group: CommGroup, recv_tensors: Tuple[torch.Tensor, ...]):
    """Wait for asynchronous communication to complete and return received tensors.

    Args:
        recv_tensors: Tuple of tensors returned from async_ring_p2p_commit
        
    Returns:
        Tuple[torch.Tensor, ...]: Tuple of received tensors after communication completes
    """
    group.p2p_wait()
    return recv_tensors
