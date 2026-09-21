// Experimental contiguous KV producer-output transport; no model integration.
#include "ulysses_group.cuh"
#include "ulysses_common.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <climits>

namespace ulysses {
namespace {
std::tuple<at::Tensor, at::Tensor, std::vector<int64_t>, std::vector<int64_t>>
kv_projection_prepare(const c10::intrusive_ptr<UlyssesGroup>& group,
    at::Tensor anchor, int64_t rows, int64_t width, std::string tag)
{
    TORCH_CHECK(anchor.is_cuda() && rows > 0 && width > 0 && width % 128 == 0,
                "invalid KV projection geometry");
    TORCH_CHECK(anchor.scalar_type() == at::kHalf || anchor.scalar_type() == at::kBFloat16,
                "KV projection requires FP16/BF16");
    const at::cuda::CUDAGuard guard(anchor.device());
    const int world = group->world_size();
    TORCH_CHECK(world == 2 || world == 4 || world == 8, "KV projection requires CP2/4/8");
    const auto& kb = group->pool().acquire({world * rows, width}, anchor.scalar_type(), tag + "_k");
    at::Tensor key = kb.view;
    std::vector<int64_t> kp(kb.peer_ptrs.begin(), kb.peer_ptrs.end());
    const auto& vb = group->pool().acquire({world * rows, width}, anchor.scalar_type(), tag + "_v");
    return {key, vb.view, kp, std::vector<int64_t>(vb.peer_ptrs.begin(), vb.peer_ptrs.end())};
}

void validate_kv_send(at::Tensor key, at::Tensor value, at::Tensor counters,
    int64_t rank, int64_t world, int64_t epoch, int64_t chunk_tokens)
{
    TORCH_CHECK((world == 2 || world == 4 || world == 8) && rank >= 0 && rank < world,
                "invalid KV rank/world");
    TORCH_CHECK(key.is_cuda() && key.dim() == 2 && key.is_contiguous()
                && key.sizes() == value.sizes() && value.is_contiguous()
                && key.device() == value.device() && key.scalar_type() == value.scalar_type()
                && (key.scalar_type() == at::kHalf || key.scalar_type() == at::kBFloat16)
                && key.size(0) > 0 && key.size(0) % world == 0
                && key.size(1) > 0 && key.size(1) % 128 == 0,
                "invalid KV symmetric outputs");
    TORCH_CHECK(chunk_tokens > 0 && chunk_tokens % 128 == 0, "invalid KV chunk");
    const int64_t rows = key.size(0) / world, width = key.size(1);
    TORCH_CHECK(counters.device() == key.device() && counters.is_contiguous()
                && counters.scalar_type() == at::kInt
                && counters.numel() == 2 * ((rows + chunk_tokens - 1) / chunk_tokens),
                "invalid KV completion counters");
    TORCH_CHECK(epoch > 0 && epoch < INT_MAX / ((chunk_tokens / 128) * (width / 128)),
                "KV completion counter exhausted");
}

void kv_projection_send_ce(at::Tensor key, at::Tensor value, at::Tensor counters,
    int64_t peer_key, int64_t peer_value, int64_t rank, int64_t world,
    int64_t epoch, int64_t chunk_tokens, bool wait_ready)
{
    validate_kv_send(key, value, counters, rank, world, epoch, chunk_tokens);
    const int64_t rows = key.size(0) / world, width = key.size(1);
    const at::cuda::CUDAGuard guard(key.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    for (int64_t row = 0; row < rows; row += chunk_tokens) {
        const int64_t count = std::min(chunk_tokens, rows - row);
        const int64_t offset = (rank * rows + row) * width * 2;
        for (int projection = 0; projection < 2; ++projection) {
            if (wait_ready) {
                const uint32_t target = epoch * ((count + 127) / 128) * (width / 128);
                CUresult status = cuStreamWaitValue32(stream,
                    reinterpret_cast<CUdeviceptr>(counters.data_ptr<int>() + 2 * (row / chunk_tokens) + projection),
                    target, CU_STREAM_WAIT_VALUE_GEQ);
                TORCH_CHECK(status == CUDA_SUCCESS, "KV CE wait-value failed: ", int(status));
            }
            auto* source = static_cast<const char*>((projection ? value : key).data_ptr());
            auto* destination = reinterpret_cast<char*>(projection ? peer_value : peer_key);
            ULYSSES_CUDA_CHECK(cudaMemcpyAsync(destination + offset, source + offset,
                count * width * 2, cudaMemcpyDefault, stream));
        }
    }
}

void kv_projection_send_ce_group(at::Tensor key, at::Tensor value, at::Tensor counters,
    std::vector<int64_t> peer_keys, std::vector<int64_t> peer_values,
    int64_t rank, int64_t world, int64_t epoch, int64_t chunk_tokens, bool wait_ready)
{
    validate_kv_send(key, value, counters, rank, world, epoch, chunk_tokens);
    TORCH_CHECK(!peer_keys.empty() && peer_keys.size() == peer_values.size()
                && peer_keys.size() < static_cast<size_t>(world), "invalid KV CE peer group");
    const at::cuda::CUDAGuard guard(key.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int64_t rows = key.size(0) / world, width = key.size(1);
    // Broadcast ready data to this stream's peers before waiting for a later chunk.
    for (int64_t row = 0; row < rows; row += chunk_tokens) {
        const int64_t count = std::min(chunk_tokens, rows - row);
        const int64_t offset = (rank * rows + row) * width * 2;
        for (int projection = 0; projection < 2; ++projection) {
            if (wait_ready) {
                const uint32_t target = epoch * ((count + 127) / 128) * (width / 128);
                CUresult status = cuStreamWaitValue32(stream,
                    reinterpret_cast<CUdeviceptr>(counters.data_ptr<int>() + 2 * (row / chunk_tokens) + projection),
                    target, CU_STREAM_WAIT_VALUE_GEQ);
                TORCH_CHECK(status == CUDA_SUCCESS, "KV grouped CE wait-value failed: ", int(status));
            }
            auto* source = static_cast<const char*>((projection ? value : key).data_ptr());
            const auto& targets = projection ? peer_values : peer_keys;
            for (auto pointer : targets) {
                ULYSSES_CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<char*>(pointer) + offset,
                    source + offset, count * width * 2, cudaMemcpyDefault, stream));
            }
        }
    }
}
} // namespace
} // namespace ulysses

TORCH_LIBRARY_FRAGMENT(fast_ulysses, m) {
    m.def("kv_projection_prepare(__torch__.torch.classes.fast_ulysses.UlyssesGroup group, Tensor anchor, int rows, int width, str tag) -> (Tensor, Tensor, int[], int[])");
    m.impl("kv_projection_prepare", c10::DispatchKey::CompositeExplicitAutograd, &ulysses::kv_projection_prepare);
    m.def("kv_projection_send_ce(Tensor key, Tensor value, Tensor counters, int peer_key, int peer_value, int rank, int world, int epoch, int chunk_tokens, bool wait_ready) -> ()");
    m.impl("kv_projection_send_ce", c10::DispatchKey::CompositeExplicitAutograd, &ulysses::kv_projection_send_ce);
    m.def("kv_projection_send_ce_group(Tensor key, Tensor value, Tensor counters, int[] peer_keys, int[] peer_values, int rank, int world, int epoch, int chunk_tokens, bool wait_ready) -> ()");
    m.impl("kv_projection_send_ce_group", c10::DispatchKey::CompositeExplicitAutograd, &ulysses::kv_projection_send_ce_group);
}
