// Experimental GEMM-output consumers; the producer is in experimental/projection.py.
#include "ulysses_group.cuh"
#include "ulysses_common.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <algorithm>
#include <climits>

namespace ulysses {
namespace {

struct ProjectionPeers { uint64_t ptr[8]; };

__global__ void projection_sync_kernel(uint64_t* local, ProjectionPeers peers, int rank, int world)
{
    __shared__ uint64_t epoch;
    if (threadIdx.x == 0) epoch = ++local[world];
    __syncthreads();
    const int peer = threadIdx.x;
    if (peer < world) {
        auto* target = reinterpret_cast<uint64_t*>(peers.ptr[peer]) + rank;
        asm volatile("st.release.sys.global.u64 [%0], %1;" :: "l"(target), "l"(epoch) : "memory");
        uint64_t seen;
        do {
            asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(seen) : "l"(local + peer) : "memory");
        } while (seen < epoch);
    }
}

std::tuple<at::Tensor, std::vector<int64_t>> projection_sync_prepare(
    const c10::intrusive_ptr<UlyssesGroup>& group, at::Tensor anchor, std::string tag)
{
    const at::cuda::CUDAGuard guard(anchor.device());
    const auto& buffer = group->pool().acquire({group->world_size() + 1}, at::kLong, tag);
    ULYSSES_CUDA_CHECK(cudaMemsetAsync(buffer.view.data_ptr(), 0,
        buffer.view.nbytes(), at::cuda::getCurrentCUDAStream()));
    group->fast_barrier(at::cuda::getCurrentCUDAStream());
    return {buffer.view, std::vector<int64_t>(buffer.peer_ptrs.begin(), buffer.peer_ptrs.end())};
}

void projection_sync(at::Tensor state, std::vector<int64_t> peers, int64_t rank)
{
    const int world = peers.size();
    TORCH_CHECK((world == 2 || world == 4 || world == 8) && rank >= 0 && rank < world,
                "invalid projection sync rank/world");
    TORCH_CHECK(state.is_cuda() && state.is_contiguous() && state.scalar_type() == at::kLong
                && state.numel() == world + 1, "invalid projection sync state");
    const at::cuda::CUDAGuard guard(state.device());
    ProjectionPeers pointers{};
    for (int i = 0; i < world; ++i) pointers.ptr[i] = peers[i];
    projection_sync_kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<uint64_t*>(state.data_ptr()), pointers, rank, world);
    ULYSSES_CUDA_CHECK(cudaGetLastError());
}

std::tuple<at::Tensor, std::vector<int64_t>> projection_prepare(
    const c10::intrusive_ptr<UlyssesGroup>& group, at::Tensor example,
    int64_t rows, int64_t width, std::string tag)
{
    TORCH_CHECK(example.is_cuda() && rows > 0 && width > 0, "invalid projection buffer");
    const int world = group->world_size();
    TORCH_CHECK(world == 2 || world == 4 || world == 8, "projection needs CP2/4/8");
    TORCH_CHECK(example.scalar_type() == at::kHalf || example.scalar_type() == at::kBFloat16,
                "projection requires FP16/BF16");
    const at::cuda::CUDAGuard guard(example.device());
    const auto& buffer = group->pool().acquire({3, world * rows, width}, example.scalar_type(), tag);
    return {buffer.view, std::vector<int64_t>(buffer.peer_ptrs.begin(), buffer.peer_ptrs.end())};
}

void projection_barrier(const c10::intrusive_ptr<UlyssesGroup>& group, at::Tensor anchor)
{
    const at::cuda::CUDAGuard guard(anchor.device());
    group->fast_barrier(at::cuda::getCurrentCUDAStream());
}

void validate(at::Tensor staging, at::Tensor flags, int64_t world, int64_t rank,
              int64_t epoch, int64_t bm, int64_t bn)
{
    TORCH_CHECK(staging.is_cuda() && staging.dim() == 2 && staging.is_contiguous(),
                "staging must be contiguous CUDA [M,3*H*D]");
    TORCH_CHECK(staging.scalar_type() == at::kHalf || staging.scalar_type() == at::kBFloat16,
                "staging must be FP16/BF16");
    TORCH_CHECK((world == 2 || world == 4 || world == 8) && rank >= 0 && rank < world,
                "invalid projection rank/world");
    TORCH_CHECK(bm == 128 && bn == 128 && staging.size(0) > 0
                    && staging.size(1) % (3 * world * bn) == 0,
                "projection tile must be 128x128, local head width a multiple of 128");
    TORCH_CHECK(epoch > 0 && epoch < INT_MAX, "invalid projection epoch");
    TORCH_CHECK(flags.device() == staging.device() && flags.scalar_type() == at::kInt
                    && flags.is_contiguous() && flags.numel() ==
                        ((staging.size(0) + bm - 1) / bm) * (staging.size(1) / bn),
                "invalid projection flags");
}

__device__ int acquire_epoch(const int* p)
{
    int value;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(p) : "memory");
    return value;
}

template<bool TRACE>
__global__ void projection_send_kernel(const uint4* staging, const int* flags,
    ProjectionPeers peers, int rows, int width, int rank, int world, int epoch,
    int64_t* times)
{
    constexpr int bm = 128, bn = 128;
    const int mt = (rows + bm - 1) / bm;
    const int nt = 3 * world * width / bn;
    const int vectors = width / 8;
    // One worker owns a complete (projection, row chunk, destination) send.
    for (int job = blockIdx.x; job < 3 * mt * (world - 1); job += gridDim.x) {
        const int remote = job % (world - 1);
        const int peer = (rank + 1 + remote) % world;
        const int m = (job / (world - 1)) % mt;
        const int projection = job / ((world - 1) * mt);
        const int n_start = (projection * world + peer) * width / bn;
        if (threadIdx.x == 0) {
            for (int n = 0; n < width / bn; ++n)
                while (acquire_epoch(flags + m * nt + n_start + n) != epoch)
                    __nanosleep(64);
        }
        __syncthreads();
        if constexpr (TRACE) {
            if (threadIdx.x == 0) {
                int64_t now;
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
                times[2 * job] = now;
            }
        }
        auto* target = reinterpret_cast<uint4*>(peers.ptr[peer]);
        const int count = min(bm, rows - m * bm) * vectors;
        for (int i = threadIdx.x; i < count; i += blockDim.x) {
            const int row = m * bm + i / vectors;
            const int col = i % vectors;
            const int64_t src = static_cast<int64_t>(row) * 3 * world * vectors
                                + (projection * world + peer) * vectors + col;
            const int64_t dst = (static_cast<int64_t>(projection) * world * rows
                                + rank * rows + row) * vectors + col;
            target[dst] = staging[src];
        }
        __threadfence_system();
        __syncthreads();
        if constexpr (TRACE) {
            if (threadIdx.x == 0) {
                int64_t now;
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
                times[2 * job + 1] = now;
            }
        }
    }
}

void projection_send_sm(at::Tensor staging, at::Tensor flags, std::vector<int64_t> peers,
                       int64_t rank, int64_t epoch, int64_t blocks, at::Tensor times)
{
    validate(staging, flags, peers.size(), rank, epoch, 128, 128);
    TORCH_CHECK(blocks > 0 && blocks <= 128, "invalid sender blocks");
    const at::cuda::CUDAGuard guard(staging.device());
    ProjectionPeers pointers{};
    for (size_t i = 0; i < peers.size(); ++i) pointers.ptr[i] = peers[i];
    TORCH_CHECK(times.device() == staging.device() && times.scalar_type() == at::kLong
                && times.is_contiguous(), "invalid sender trace buffer");
    if (times.numel()) {
        TORCH_CHECK(times.numel() == 6 * ((staging.size(0) + 127) / 128) * (peers.size() - 1),
                    "invalid sender trace shape");
        projection_send_kernel<true><<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const uint4*>(staging.data_ptr()), flags.data_ptr<int>(), pointers,
            staging.size(0), staging.size(1) / (3 * peers.size()), rank, peers.size(), epoch,
            times.data_ptr<int64_t>());
    } else {
        projection_send_kernel<false><<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const uint4*>(staging.data_ptr()), flags.data_ptr<int>(), pointers,
            staging.size(0), staging.size(1) / (3 * peers.size()), rank, peers.size(), epoch, nullptr);
    }
    ULYSSES_CUDA_CHECK(cudaGetLastError());
}

void projection_send_ce(at::Tensor staging, at::Tensor flags, int64_t peer_ptr,
                       int64_t peer, int64_t rank, int64_t world, int64_t epoch,
                       int64_t chunk_tokens, int64_t row_start, int64_t row_count,
                       bool wait_tiles)
{
    validate(staging, flags, world, rank, epoch, 128, 128);
    TORCH_CHECK(peer >= 0 && peer < world && peer != rank, "invalid remote peer");
    TORCH_CHECK(chunk_tokens > 0 && chunk_tokens % 128 == 0,
                "CE chunk_tokens must be a positive multiple of 128");
    TORCH_CHECK(row_start >= 0 && row_start % 128 == 0 && row_count > 0
                && row_start + row_count <= staging.size(0), "invalid CE row range");
    const at::cuda::CUDAGuard guard(staging.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int64_t rows = staging.size(0), n = staging.size(1), width = n / (3 * world);
    const int64_t nt = n / 128;
    for (int projection = 0; projection < 3; ++projection) {
        for (int64_t row = row_start; row < row_start + row_count; row += chunk_tokens) {
            const int64_t count = std::min(chunk_tokens, row_start + row_count - row);
            const int64_t n_start = (projection * world + peer) * width / 128;
            if (wait_tiles) {
                for (int64_t m = row / 128; m < (row + count + 127) / 128; ++m) {
                    for (int tile = 0; tile < width / 128; ++tile) {
                        CUresult status = cuStreamWaitValue32(stream,
                            reinterpret_cast<CUdeviceptr>(flags.data_ptr<int>() + m * nt + n_start + tile),
                            epoch, CU_STREAM_WAIT_VALUE_EQ);
                        TORCH_CHECK(status == CUDA_SUCCESS, "CE wait-value failed: ", int(status));
                    }
                }
            }
            const int64_t src = (row * n + (projection * world + peer) * width) * 2;
            const int64_t dst = (projection * world * rows + rank * rows + row) * width * 2;
            ULYSSES_CUDA_CHECK(cudaMemcpy2DAsync(
                reinterpret_cast<char*>(peer_ptr) + dst, width * 2,
                static_cast<const char*>(staging.data_ptr()) + src, n * 2,
                width * 2, count, cudaMemcpyDefault, stream));
        }
    }
}

} // namespace
void projection_sync_device(at::Tensor state, std::vector<int64_t> peers, int64_t rank)
{
    projection_sync(state, peers, rank);
}
} // namespace ulysses

TORCH_LIBRARY_FRAGMENT(fast_ulysses, m) {
    m.def("projection_sync_prepare(__torch__.torch.classes.fast_ulysses.UlyssesGroup group, Tensor anchor, str tag) -> (Tensor, int[])");
    m.impl("projection_sync_prepare", c10::DispatchKey::CompositeExplicitAutograd, &ulysses::projection_sync_prepare);
    m.def("projection_sync(Tensor state, int[] peers, int rank) -> ()");
    m.impl("projection_sync", c10::DispatchKey::CompositeExplicitAutograd, &ulysses::projection_sync);
    m.def("projection_prepare(__torch__.torch.classes.fast_ulysses.UlyssesGroup group, "
          "Tensor example, int rows, int width, str tag) -> (Tensor, int[])");
    m.impl("projection_prepare", c10::DispatchKey::CompositeExplicitAutograd, &ulysses::projection_prepare);
    m.def("projection_barrier(__torch__.torch.classes.fast_ulysses.UlyssesGroup group, Tensor anchor) -> ()");
    m.impl("projection_barrier", c10::DispatchKey::CompositeExplicitAutograd, &ulysses::projection_barrier);
    m.def("projection_send_sm(Tensor staging, Tensor flags, int[] peers, int rank, int epoch, int blocks, Tensor times) -> ()");
    m.impl("projection_send_sm", c10::DispatchKey::CompositeExplicitAutograd, &ulysses::projection_send_sm);
    m.def("projection_send_ce(Tensor staging, Tensor flags, int peer_ptr, int peer, int rank, int world, int epoch, "
          "int chunk_tokens, int row_start, int row_count, bool wait_tiles) -> ()");
    m.impl("projection_send_ce", c10::DispatchKey::CompositeExplicitAutograd, &ulysses::projection_send_ce);
}
