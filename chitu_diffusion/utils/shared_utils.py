import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional

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

def split_latent(shape: torch.Size, split_num: int, split_dim = 1) -> List[Tuple[int,int]]:
    
    full_size = shape[split_dim]
    split_size = full_size // split_num
    remainder = full_size % split_num

    split_start_end_idxs = []
    
    start_idx = 0
    for i in range(split_num):
        end_idx = start_idx + split_size
        if i < remainder:  
            end_idx += 1
        split_start_end_idxs.append((start_idx, end_idx))
        start_idx = end_idx
    
    return split_start_end_idxs





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
    out: Optional[torch.Tensor], # （b, s, a, d）
    lse: Optional[torch.Tensor], # (b, s, d, 1)
    block_out: torch.Tensor, # (b, s, a, d)
    block_lse: torch.Tensor, # (b, d, s)
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    # output: out (b, s, a, d) lse (b, s, d, 1)        
    if out is None:
      out = block_out.to(torch.float32)
      lse = transpose_and_unsqueeze(block_lse)
    else:
      out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
      
    return out, lse