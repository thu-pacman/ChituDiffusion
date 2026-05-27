from dataclasses import dataclass
from typing import Optional

import torch

from chitu_diffusion.runtime.parallel_utils import squeeze_and_transpose, update_out_and_lse


@dataclass
class AttentionState:
    out: Optional[torch.Tensor] = None
    lse: Optional[torch.Tensor] = None

    def is_empty(self) -> bool:
        return self.lse is None

    def update(self, block_out: torch.Tensor, block_lse: torch.Tensor):
        self.out, self.lse = update_out_and_lse(self.out, self.lse, block_out, block_lse)

    @staticmethod
    def merge(state1: "AttentionState", state2: "AttentionState") -> "AttentionState":
        if state2.is_empty():
            return state1
        input_lse = squeeze_and_transpose(state2.lse)
        state1.update(state2.out, input_lse)
        return state1


__all__ = ["AttentionState"]
