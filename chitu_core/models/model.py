# TODO: Unified Diffusion Model Architecture.
# 来个好心人把这伟大的活contribute了吧。
from dataclasses import dataclass
import torch
    

@dataclass
class DITParam:
    '''
        time_proj (Tensor): Shape [B, 6, C]
        context_embedding (Tensor): Shape [B, L, C]
        grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
        freqs_params(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        context_lens(Tensor): Shape [B], the length of context for each sample in the batch.
    '''
    time_proj : torch.Tensor
    context_embedding : torch.Tensor
    grid_sizes : torch.Tensor
    freq_params : torch.Tensor # the pre_compute tensor for rope operation
    context_lens = None

@dataclass
class PostDITParams:
    '''
        tokens (Tensor): Shape [B, L, C]
        time_embedding (Tensor): Shape [B, C]
        grid_sizes (Tensor): Shape [B, 3], the second dimension contains (F, H, W)
    '''
    time_embedding : torch.Tensor
    grid_sizes : torch.Tensor