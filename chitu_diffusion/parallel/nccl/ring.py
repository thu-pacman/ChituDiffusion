from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ..topology import UspTopology


def gather_replicated_heads(
    tensor: torch.Tensor,
    topology: UspTopology,
) -> torch.Tensor:
    if topology.ulysses_degree == 1:
        return tensor
    if tensor.shape[1] == 0:
        return tensor.new_empty(
            tensor.shape[0],
            0,
            tensor.shape[2] * topology.ulysses_degree,
            tensor.shape[3],
        )
    pieces = [torch.empty_like(tensor) for _ in range(topology.ulysses_degree)]
    dist.all_gather(
        pieces,
        tensor.contiguous(),
        group=topology.ulysses_process_group,
    )
    return torch.cat(pieces, dim=2).contiguous()


def _local_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        dropout_p=0.0,
        is_causal=False,
    ).transpose(1, 2)


class _RingComm:
    def __init__(self, process_group: object) -> None:
        self.process_group = process_group
        self.rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)
        send_local = (self.rank + 1) % self.world_size
        recv_local = (self.rank - 1) % self.world_size
        self.send_rank = dist.get_global_rank(process_group, send_local)
        self.recv_rank = dist.get_global_rank(process_group, recv_local)
        self._ops: list[dist.P2POp] = []
        self._requests = None

    def send_recv(self, tensor: torch.Tensor) -> torch.Tensor:
        received = torch.empty_like(tensor)
        self._ops.extend(
            (
                dist.P2POp(
                    dist.isend,
                    tensor,
                    self.send_rank,
                    group=self.process_group,
                ),
                dist.P2POp(
                    dist.irecv,
                    received,
                    self.recv_rank,
                    group=self.process_group,
                ),
            )
        )
        return received

    def commit(self) -> None:
        if self._requests is not None:
            raise RuntimeError("ring communication was committed twice")
        self._requests = dist.batch_isend_irecv(self._ops)

    def wait(self) -> None:
        if self._requests is None:
            raise RuntimeError("ring communication wait called before commit")
        for request in self._requests:
            request.wait()
        self._requests = None
        self._ops = []


def _merge_attention_blocks(
    output: torch.Tensor | None,
    output_lse: torch.Tensor | None,
    block_output: torch.Tensor,
    block_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_output = block_output.float()
    block_lse = block_lse.transpose(-2, -1).unsqueeze(-1)
    if output is None or output_lse is None:
        return block_output, block_lse
    output = output - F.sigmoid(block_lse - output_lse) * (output - block_output)
    output_lse = output_lse - F.logsigmoid(output_lse - block_lse)
    return output, output_lse


def joint_ring_attention(
    query: torch.Tensor,
    image_key: torch.Tensor,
    image_value: torch.Tensor,
    joint_key: torch.Tensor,
    joint_value: torch.Tensor,
    topology: UspTopology,
) -> torch.Tensor:
    if topology.ring_degree == 1:
        return _local_attention(
            query,
            torch.cat([image_key, joint_key], dim=1),
            torch.cat([image_value, joint_value], dim=1),
        )
    if not query.is_cuda:
        key_pieces = [torch.empty_like(image_key) for _ in range(topology.ring_degree)]
        value_pieces = [
            torch.empty_like(image_value) for _ in range(topology.ring_degree)
        ]
        dist.all_gather(
            key_pieces,
            image_key.contiguous(),
            group=topology.ring_process_group,
        )
        dist.all_gather(
            value_pieces,
            image_value.contiguous(),
            group=topology.ring_process_group,
        )
        return _local_attention(
            query,
            torch.cat([*key_pieces, joint_key], dim=1),
            torch.cat([*value_pieces, joint_value], dim=1),
        )

    from flash_attn import flash_attn_func

    comm = _RingComm(topology.ring_process_group)
    key = image_key.contiguous()
    value = image_value.contiguous()
    output = None
    output_lse = None
    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_key = comm.send_recv(key)
            next_value = comm.send_recv(value)
            comm.commit()
        if step + 1 == comm.world_size:
            block_key = torch.cat([key, joint_key], dim=1)
            block_value = torch.cat([value, joint_value], dim=1)
        else:
            block_key = key
            block_value = value
        block_output, block_lse, _ = flash_attn_func(
            query,
            block_key,
            block_value,
            dropout_p=0.0,
            softmax_scale=None,
            causal=False,
            window_size=(-1, -1),
            softcap=0.0,
            alibi_slopes=None,
            return_attn_probs=True,
        )
        output, output_lse = _merge_attention_blocks(
            output,
            output_lse,
            block_output,
            block_lse,
        )
        if step + 1 != comm.world_size:
            comm.wait()
            key = next_key
            value = next_value
    return output.to(query.dtype)
