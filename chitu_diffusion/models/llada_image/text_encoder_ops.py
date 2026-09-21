from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def fused_moe_forward(
    module: nn.Module,
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_weight: torch.Tensor,
    fc1_2_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
) -> torch.Tensor:
    """Evaluate routed SwiGLU experts."""

    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if any(
        weight.shape[0] != num_experts
        for weight in (fc1_1_weight, fc1_2_weight, fc2_weight)
    ):
        raise ValueError("all expert weight tensors must contain num_experts experts")
    original_shape = hidden_states.shape
    flat_hidden = hidden_states.reshape(-1, original_shape[-1])
    flat_experts = selected_experts.reshape(flat_hidden.shape[0], -1).to(
        dtype=torch.long
    )
    flat_routing_weights = routing_weights.reshape(flat_hidden.shape[0], -1)
    if flat_experts.shape != flat_routing_weights.shape:
        raise ValueError(
            "selected experts and routing weights must have the same shape"
        )
    if flat_experts.numel() and (
        flat_experts.min() < 0 or flat_experts.max() >= num_experts
    ):
        raise ValueError("selected expert indices are out of range")

    activation = getattr(module, "act_fn", F.silu)
    if not callable(activation):
        raise TypeError("module.act_fn must be callable")
    output = torch.zeros_like(flat_hidden)
    for expert_index in range(num_experts):
        token_indices, route_indices = torch.where(flat_experts == expert_index)
        if token_indices.numel() == 0:
            continue
        expert_input = flat_hidden[token_indices]
        gate = F.linear(expert_input, fc1_1_weight[expert_index])
        up = F.linear(expert_input, fc1_2_weight[expert_index])
        expert_output = F.linear(activation(gate) * up, fc2_weight[expert_index])
        route_weight = flat_routing_weights[token_indices, route_indices, None]
        output.index_add_(
            0,
            token_indices,
            expert_output * route_weight.to(expert_output.dtype),
        )
    return output.view(original_shape)
