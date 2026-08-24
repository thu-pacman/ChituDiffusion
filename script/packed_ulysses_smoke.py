from __future__ import annotations

import os

import torch
import torch.distributed as dist

from chitu_diffusion.parallel import (
    EpeParallelContext,
    SdpaVarlenBackend,
    all_to_all_packed_output,
    all_to_all_packed_qkv,
)


def main() -> None:
    context = EpeParallelContext.from_torchrun(
        allowed_widths=(1, 2),
        ulysses_degree=2,
    )
    rank = int(os.environ["RANK"])
    generator = torch.Generator(device="cuda").manual_seed(91)
    q = torch.randn(8, 4, 8, generator=generator, device="cuda")
    k = torch.randn(8, 4, 8, generator=generator, device="cuda")
    v = torch.randn(8, 4, 8, generator=generator, device="cuda")
    local = slice(rank * 4, (rank + 1) * 4)
    q_global, k_global, v_global = all_to_all_packed_qkv(
        q[local], k[local], v[local], topology=context.active_usp
    )
    cu = torch.tensor([0, 6, 8], dtype=torch.int32, device="cuda")
    local_heads = slice(rank * 2, (rank + 1) * 2)
    torch.testing.assert_close(q_global, q[:, local_heads])
    output = SdpaVarlenBackend().forward_varlen(
        q_global,
        k_global,
        v_global,
        cu_seqlens=cu,
        max_seqlen=6,
    )
    restored = all_to_all_packed_output(output, topology=context.active_usp)
    expected = SdpaVarlenBackend().forward_varlen(
        q, k, v, cu_seqlens=cu, max_seqlen=6
    )
    torch.testing.assert_close(restored, expected[local], rtol=1e-5, atol=1e-5)
    if rank == 0:
        print("packed Ulysses CUDA parity passed")
    context.close()


if __name__ == "__main__":
    main()

