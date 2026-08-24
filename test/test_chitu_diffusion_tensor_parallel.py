from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from packaging.version import Version

from chitu_diffusion.models.minimax_h3 import (
    MiniMaxH3DiTConfig,
    MiniMaxH3DiTModel,
    build_packed_sequence,
    reorder_grouped_qkv_to_qkv,
)
from chitu_diffusion.parallel import (
    ColumnParallelLinear,
    EpeParallelContext,
    MergedColumnParallelLinear,
    RowParallelLinear,
    SdpaVarlenBackend,
    all_to_all_packed_output,
    all_to_all_packed_qkv,
)
from chitu_diffusion.parallel.attention_backend import (
    Fa4VarlenBackend,
    FlexVarlenBackend,
    _fa4_supports_compute_capability,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _initialize_worker(rank: int, world_size: int, port: int) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
    )
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def _tp_linear_worker(rank: int, world_size: int, port: int) -> None:
    _initialize_worker(rank, world_size, port)
    context = EpeParallelContext.from_torchrun(
        allowed_widths=(1,),
        tensor_parallel_degree=world_size,
        owns_process_group=False,
    )
    generator = torch.Generator().manual_seed(17)
    inputs = torch.randn(3, 8, generator=generator)

    column_weight = torch.randn(12, 8, generator=generator)
    column_bias = torch.randn(12, generator=generator)
    column = ColumnParallelLinear(8, 12, gather_output=True)
    column.weight_loader(column.weight, column_weight)
    column.weight_loader(column.bias, column_bias)
    column_output, _ = column(inputs)
    torch.testing.assert_close(
        column_output, F.linear(inputs, column_weight, column_bias)
    )

    row_weight = torch.randn(6, 8, generator=generator)
    row_bias = torch.randn(6, generator=generator)
    row = RowParallelLinear(8, 6, input_is_parallel=False)
    row.weight_loader(row.weight, row_weight)
    row.weight_loader(row.bias, row_bias)
    row_output, _ = row(inputs)
    torch.testing.assert_close(row_output, F.linear(inputs, row_weight, row_bias))

    merged_weight = torch.randn(16, 8, generator=generator)
    merged = MergedColumnParallelLinear(
        8, (6, 10), bias=False, gather_output=True
    )
    merged.weight_loader(merged.weight, merged_weight)
    merged_output, _ = merged(inputs)
    torch.testing.assert_close(merged_output, F.linear(inputs, merged_weight))

    dist.barrier()
    context.close()
    dist.destroy_process_group()


def _packed_ulysses_worker(rank: int, world_size: int, port: int) -> None:
    _initialize_worker(rank, world_size, port)
    context = EpeParallelContext.from_torchrun(
        allowed_widths=(1, 2),
        ulysses_degree=2,
        owns_process_group=False,
    )
    generator = torch.Generator().manual_seed(23 + rank)
    query = torch.randn(4, 4, 3, generator=generator)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    topology = context.active_usp
    global_query, global_key, global_value = all_to_all_packed_qkv(
        query, key, value, topology=topology
    )
    assert global_query.shape == global_key.shape == global_value.shape == (8, 2, 3)
    restored = all_to_all_packed_output(global_query, topology=topology)
    torch.testing.assert_close(restored, query)
    dist.barrier()
    context.close()
    dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_tensor_parallel_linear_matches_dense() -> None:
    mp.spawn(_tp_linear_worker, args=(2, _free_port()), nprocs=2, join=True)


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_packed_ulysses_round_trip() -> None:
    mp.spawn(_packed_ulysses_worker, args=(2, _free_port()), nprocs=2, join=True)


def test_sdpa_varlen_documents_match_independent_attention() -> None:
    generator = torch.Generator().manual_seed(29)
    query = torch.randn(7, 2, 4, generator=generator)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    cu = torch.tensor([0, 5, 7], dtype=torch.int32)
    output = SdpaVarlenBackend().forward_varlen(
        query, key, value, cu_seqlens=cu, max_seqlen=5
    )
    expected = torch.cat(
        [
            F.scaled_dot_product_attention(
                query[start:stop].transpose(0, 1),
                key[start:stop].transpose(0, 1),
                value[start:stop].transpose(0, 1),
            ).transpose(0, 1)
            for start, stop in ((0, 5), (5, 7))
        ]
    )
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize(
    ("capability", "version", "expected"),
    [
        ((9, 0), None, True),
        ((9, 0), Version("4.0.0b19"), True),
        ((12, 0), None, False),
        ((12, 0), Version("4.0.0b19"), False),
        ((12, 0), Version("4.0.0b25"), True),
        ((12, 0), Version("4.0.0b26"), True),
        ((12, 0), Version("4.0.0"), True),
    ],
)
def test_fa4_sm120_version_gate(
    capability: tuple[int, int],
    version: Version | None,
    expected: bool,
) -> None:
    assert _fa4_supports_compute_capability(capability, version) is expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_flex_varlen_matches_sdpa() -> None:
    generator = torch.Generator(device="cuda").manual_seed(31)
    query = torch.randn(
        64, 2, 128, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    cu = torch.tensor([0, 48, 64], device="cuda", dtype=torch.int32)
    expected = SdpaVarlenBackend().forward_varlen(
        query, key, value, cu_seqlens=cu, max_seqlen=48
    )
    output = FlexVarlenBackend().forward_varlen(
        query, key, value, cu_seqlens=cu, max_seqlen=48
    )
    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_fa4_varlen_matches_sdpa_when_installed() -> None:
    try:
        from flash_attn.cute.interface import flash_attn_varlen_func  # noqa: F401
    except ImportError:
        pytest.skip("flash-attn-4 is not installed")
    query = torch.randn(64, 2, 128, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0, 48, 64], device="cuda", dtype=torch.int32)
    expected = SdpaVarlenBackend().forward_varlen(
        query, query, query, cu_seqlens=cu, max_seqlen=48
    )
    output = Fa4VarlenBackend().forward_varlen(
        query, query, query, cu_seqlens=cu, max_seqlen=48
    )
    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)


def test_h3_grouped_qkv_reorder() -> None:
    source = torch.arange(4 * 3 * 2 * 3).reshape(4 * 3 * 2, 3)
    output = reorder_grouped_qkv_to_qkv(source, num_heads=4, head_dim=2)
    grouped = source.reshape(4, 3, 2, 3)
    expected = grouped.permute(1, 0, 2, 3).reshape(24, 3)
    torch.testing.assert_close(output, expected)


def test_h3_packed_sequence_and_tiny_forward() -> None:
    layout = build_packed_sequence(
        text_length=2,
        latent_t=1,
        latent_h=4,
        latent_w=4,
        audio_t=1,
    )
    assert layout.sequence_length == 64
    assert layout.cu_seqlens.tolist() == [0, 8, 64]
    config = MiniMaxH3DiTConfig(
        hidden_size=16,
        num_layers=1,
        token_refiner_num_layers=1,
        num_attention_heads=2,
        attention_head_dim=8,
        ffn_hidden_size=32,
        latents_dim=2,
        audio_latents_dim=4,
        patch_size=(1, 1, 1),
        text_dim=12,
        timestep_input_dim=8,
        time_embed_hidden_size=16,
        time_embed_dim=8,
        adaln_out_features=18 * 16,
        final_adaln_out_features=2 * 16,
        rope_inv_freq_len=1,
    )
    model = MiniMaxH3DiTModel(config, attention_backend="sdpa")
    hidden = torch.randn(8, 16, dtype=torch.bfloat16)
    video, audio = model.forward_hidden(
        hidden,
        unique_timesteps=torch.tensor([1.0]),
        inverse_indices=torch.zeros(8, dtype=torch.long),
        token_tags=torch.tensor([1, 1, 2, 2, 0, 0, 0, -1]),
        position_ids=torch.zeros(8, 3, dtype=torch.float64),
        cu_seqlens=torch.tensor([0, 7, 8], dtype=torch.int32),
        max_seqlen=7,
    )
    assert video.shape == (8, 2)
    assert audio.shape == (8, 4)
    assert torch.isfinite(video).all()
    assert torch.isfinite(audio).all()

