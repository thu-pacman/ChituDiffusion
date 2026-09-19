"""SmartDiffusion's equal CP stripes and equal padded FPP patches."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FppTokenLayout:
    tokens: int
    patches: int
    context_degree: int = 1

    def __post_init__(self):
        if any(
            type(n) is not int or n < 1
            for n in (self.tokens, self.patches, self.context_degree)
        ):
            raise ValueError(
                "FPP token, patch and context counts must be positive integers"
            )

    @property
    def patch_tokens(self):
        unit = self.patches * self.context_degree
        return (self.tokens + unit - 1) // unit

    @property
    def local_tokens(self):
        return self.patch_tokens * self.patches

    @property
    def padded_tokens(self):
        return self.local_tokens * self.context_degree

    def indices(self, patch: int | None, cp_rank: int | None, *, device):
        """cp_rank=None emulates every CP stripe on one reference device."""
        start, stop = (
            (0, self.local_tokens)
            if patch is None
            else (patch * self.patch_tokens, (patch + 1) * self.patch_tokens)
        )
        ranks = range(self.context_degree) if cp_rank is None else (cp_rank,)
        return torch.cat(
            [
                torch.arange(
                    c * self.local_tokens + start,
                    c * self.local_tokens + stop,
                    device=device,
                )
                for c in ranks
            ]
        )
