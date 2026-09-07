#include "symmetric_pool.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <torch/torch.h>

namespace ulysses {

SymmetricHeapPool::SymmetricHeapPool(
    int64_t reserved_bytes, int world_size, std::vector<int> peer_global_pes):
    SymmetricHeapPool(
        nvshmem_align(256, static_cast<size_t>(reserved_bytes)),
        reserved_bytes,
        world_size,
        std::move(peer_global_pes))
{
    owns_arena_ = true;
}

SymmetricHeapPool::SymmetricHeapPool(
    void* arena_base, int64_t reserved_bytes, int world_size, std::vector<int> peer_global_pes):
    arena_base_(arena_base),
    reserved_(reserved_bytes),
    world_size_(world_size),
    peer_global_pes_(std::move(peer_global_pes))
{
    TORCH_CHECK(arena_base_ != nullptr, "SymmetricHeapPool requires an initialized symmetric arena");
}

const SymmetricHeapPool::Buffer&
SymmetricHeapPool::acquire(const std::vector<int64_t>& shape,
                           c10::ScalarType             dtype,
                           const std::string&          tag,
                           bool*                       created)
{
    TORCH_CHECK(!destroyed_, "SymmetricHeapPool::acquire called after destroy()");
    if (created)
        *created = false;
    Key  key{tag, shape, dtype};
    auto it = registry_.find(key);
    if (it != registry_.end())
        return it->second;
    if (created)
        *created = true;

    int64_t numel = 1;
    for (auto s : shape)
        numel *= s;
    const int64_t elem   = c10::elementSize(dtype);
    int64_t       nbytes = numel * elem;
    nbytes               = (nbytes + 15) / 16 * 16;

    const int64_t alloc_bytes = nbytes;
    const int64_t offset      = (used_ + 255) / 256 * 256;
    TORCH_CHECK(offset + alloc_bytes <= reserved_,
                "SymmetricHeapPool OOM: need ",
                alloc_bytes,
                " B, used ",
                used_,
                " / reserved ",
                reserved_,
                " B. Increase initial_pool_bytes.");

    void* p = static_cast<void*>(static_cast<char*>(arena_base_) + offset);
    used_   = offset + alloc_bytes;

    Buffer buf;
    buf.sym_base = p;
    buf.nbytes   = alloc_bytes;
    buf.peer_ptrs.resize(world_size_);
    for (int i = 0; i < world_size_; ++i)
        buf.peer_ptrs[i] = reinterpret_cast<uint64_t>(nvshmem_ptr(p, peer_global_pes_[i]));
    for (int i = 0; i < world_size_; ++i)
        TORCH_CHECK(buf.peer_ptrs[i] != 0,
                    "nvshmem_ptr returned NULL for peer ",
                    i,
                    " (non-P2P-reachable; Fast AGKV requires a single-node P2P group).");

    auto opts = at::TensorOptions().dtype(dtype).device(at::kCUDA, at::cuda::current_device());
    buf.view  = at::from_blob(p, shape, [](void*) {}, opts);

    auto res = registry_.emplace(std::move(key), std::move(buf));
    return res.first->second;
}

void SymmetricHeapPool::destroy()
{
    if (destroyed_)
        return;
    registry_.clear();
    if (owns_arena_ && arena_base_ != nullptr) {
        nvshmem_free(arena_base_);
        arena_base_ = nullptr;
    }
    destroyed_ = true;
}

}  // namespace ulysses
