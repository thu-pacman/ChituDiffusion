"""Variable-length all-to-all dispatch and combine for routed experts."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.distributed as dist

from .topology import ExpertParallelTopology

RunExpert = Callable[[int, torch.Tensor], torch.Tensor]


def _exchange_count_table(
    counts: torch.Tensor,
    topology: ExpertParallelTopology,
) -> torch.Tensor:
    """Exchange a ``[degree, columns]`` table, one row per peer."""

    received = torch.empty_like(counts)
    dist.all_to_all_single(received, counts.contiguous(), group=topology.process_group)
    return received


def _block_offsets(counts: torch.Tensor) -> torch.Tensor:
    """Exclusive prefix sum: where each peer's block starts in the buffer."""

    return torch.cumsum(counts, 0) - counts


def _histogram(
    values: torch.Tensor, size: int, weights: torch.Tensor | None = None
) -> torch.Tensor:
    """Count ``values`` into ``size`` buckets without reading the device.

    ``torch.bincount`` reduces the maximum on the device and copies it back to
    size its own output, which stalls the queue once per call. The bucket count
    is known here, so a scatter-add keeps the whole histogram asynchronous.
    """

    counts = torch.zeros(size, dtype=torch.int64, device=values.device)
    if weights is None:
        weights = torch.ones_like(values, dtype=torch.int64)
    return counts.index_add_(0, values, weights.to(torch.int64))


def _apply_owned_experts(
    rows: torch.Tensor,
    pair_row: torch.Tensor,
    pair_expert: torch.Tensor,
    pair_weight: torch.Tensor,
    *,
    topology: ExpertParallelTopology,
    run_expert: RunExpert,
) -> torch.Tensor:
    """Run this rank's experts and fold their weighted outputs back into ``rows``.

    Each routed pair names a row and one of this rank's experts, so a row shared
    by several experts is gathered once per pair but stored once. Sorting the
    pairs by expert and composing that order into the gather keeps a single pass
    over the hidden states in each direction.

    The segment sizes have to reach the host to slice the buffer, so ownership
    is validated in that same transfer rather than by reading the device again.
    """

    reduced = torch.zeros_like(rows)
    if pair_row.shape[0] == 0:
        return reduced
    local_ids = pair_expert - topology.local_expert_start
    owned = (local_ids >= 0) & (local_ids < topology.experts_per_rank)
    bucket = torch.where(owned, local_ids, torch.zeros_like(local_ids))
    order = torch.argsort(bucket, stable=True)
    sorted_row = pair_row.index_select(0, order)
    sorted_rows = rows.index_select(0, sorted_row)
    table = torch.cat(
        [
            _histogram(bucket, topology.experts_per_rank),
            (~owned).sum().reshape(1),
        ]
    ).tolist()
    if table[-1]:
        raise ValueError("received a token routed to an expert this rank does not own")
    counts = table[:-1]
    outputs = torch.empty_like(sorted_rows)
    offset = 0
    for local_index, count in enumerate(counts):
        if count:
            expert_index = topology.local_expert_start + local_index
            segment = sorted_rows[offset : offset + count]
            result = run_expert(expert_index, segment)
            if result.shape != segment.shape:
                raise ValueError(
                    f"expert {expert_index} returned {tuple(result.shape)}, "
                    f"expected {tuple(segment.shape)}"
                )
            outputs[offset : offset + count] = result
        offset += count
    outputs *= (
        pair_weight.index_select(0, order).unsqueeze(-1).to(outputs.dtype)
    )
    return reduced.index_add_(0, sorted_row, outputs)


def expert_parallel_moe(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    topology: ExpertParallelTopology,
    run_expert: RunExpert,
) -> torch.Tensor:
    """Route flat tokens to their owning experts and combine the results.

    ``hidden_states`` is ``[tokens, hidden]``, ``topk_weights`` and
    ``topk_indices`` are ``[tokens, topk]``.

    A token whose routed experts share an owner crosses the wire once rather
    than once per expert: rows are deduplicated per destination, and the routed
    pairs travel beside them as indices into the row block. The owning rank
    therefore applies the router weights and sums its own experts before the
    inverse exchange, and the source only adds up the one row it gets back from
    each destination.
    """

    if hidden_states.ndim != 2:
        raise ValueError("expert dispatch requires flat [tokens, hidden] states")
    if topk_indices.shape != topk_weights.shape:
        raise ValueError("router weights and indices must have the same shape")
    if topk_indices.shape[0] != hidden_states.shape[0]:
        raise ValueError("router decisions must cover every token")

    tokens, hidden_size = hidden_states.shape
    topk = topk_indices.shape[1]
    flat_indices = topk_indices.reshape(-1).to(torch.int64)
    flat_weights = topk_weights.reshape(-1)
    token_index = torch.arange(
        tokens, device=hidden_states.device
    ).repeat_interleave(topk)

    if topology.degree == 1:
        return _apply_owned_experts(
            hidden_states,
            token_index,
            flat_indices,
            flat_weights,
            topology=topology,
            run_expert=run_expert,
        )

    destinations = topology.owner_of_tensor(flat_indices)
    order = torch.argsort(destinations, stable=True)
    sorted_destinations = destinations.index_select(0, order)
    sorted_tokens = token_index.index_select(0, order)

    # The router emits its pairs token by token, and a stable sort by
    # destination keeps that order inside each peer's block. Pairs that name the
    # same row are therefore adjacent, and the rows to send are just the
    # boundaries between them: no second sort, and no data-dependent shape.
    starts = torch.ones_like(sorted_tokens, dtype=torch.bool)
    starts[1:] = (sorted_tokens[1:] != sorted_tokens[:-1]) | (
        sorted_destinations[1:] != sorted_destinations[:-1]
    )
    pair_row = torch.cumsum(starts, 0) - 1
    send_rows = _histogram(sorted_destinations, topology.degree, starts)
    send_pairs = _histogram(sorted_destinations, topology.degree)
    send_pair_row = pair_row - _block_offsets(send_rows).index_select(
        0, sorted_destinations
    )

    # Both count tables cross to the host together, because every device read
    # drains the queue and the splits are the only values the collectives need
    # in Python.
    sent_counts = torch.stack([send_rows, send_pairs], dim=1)
    received_counts = _exchange_count_table(sent_counts, topology)
    splits = torch.cat([sent_counts, received_counts], dim=1).tolist()
    send_row_splits = [row[0] for row in splits]
    send_pair_splits = [row[1] for row in splits]
    recv_row_splits = [row[2] for row in splits]
    recv_pair_splits = [row[3] for row in splits]

    # Every pair writes its own row index, so the duplicates a shared row
    # produces all store the same token.
    send_token_index = sorted_tokens.new_zeros(sum(send_row_splits))
    send_token_index.scatter_(0, pair_row, sorted_tokens)
    send_hidden = hidden_states.index_select(0, send_token_index)

    recv_hidden = send_hidden.new_empty((sum(recv_row_splits), hidden_size))
    dist.all_to_all_single(
        recv_hidden,
        send_hidden.contiguous(),
        output_split_sizes=recv_row_splits,
        input_split_sizes=send_row_splits,
        group=topology.process_group,
    )
    # Row index and expert id share one exchange; both are small non-negative
    # integers, so the pair packs into a single value.
    packed = send_pair_row * topology.num_experts + flat_indices.index_select(0, order)
    recv_packed = packed.new_empty(sum(recv_pair_splits))
    recv_weights = flat_weights.new_empty(sum(recv_pair_splits))
    for output, payload in (
        (recv_packed, packed),
        (recv_weights, flat_weights.index_select(0, order)),
    ):
        dist.all_to_all_single(
            output,
            payload.contiguous(),
            output_split_sizes=recv_pair_splits,
            input_split_sizes=send_pair_splits,
            group=topology.process_group,
        )
    recv_pair_row = torch.div(recv_packed, topology.num_experts, rounding_mode="floor")
    recv_experts = recv_packed - recv_pair_row * topology.num_experts
    # Each peer indexed into its own block; rebase onto the concatenated buffer.
    # The expansion length is already known on the host, so it is passed in
    # rather than recovered from the device.
    recv_pair_row = recv_pair_row + torch.repeat_interleave(
        _block_offsets(received_counts[:, 0]),
        received_counts[:, 1],
        output_size=sum(recv_pair_splits),
    )

    reduced = _apply_owned_experts(
        recv_hidden,
        recv_pair_row,
        recv_experts,
        recv_weights,
        topology=topology,
        run_expert=run_expert,
    )

    returned = torch.empty_like(send_hidden)
    dist.all_to_all_single(
        returned,
        reduced.contiguous(),
        output_split_sizes=send_row_splits,
        input_split_sizes=recv_row_splits,
        group=topology.process_group,
    )
    combined = torch.zeros_like(hidden_states)
    return combined.index_add_(0, send_token_index, returned)
