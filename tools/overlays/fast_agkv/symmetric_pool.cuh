#pragma once
#include <ATen/ATen.h>
#include <cstdint>
#include <map>
#include <nvshmem.h>
#include <string>
#include <tuple>
#include <vector>

namespace ulysses {

class SymmetricHeapPool {
public:
    SymmetricHeapPool(int64_t reserved_bytes, int world_size, std::vector<int> peer_global_pes);
    SymmetricHeapPool(void* arena_base, int64_t reserved_bytes, int world_size, std::vector<int> peer_global_pes);

    struct Buffer {
        void*                 sym_base;
        int64_t               nbytes;
        std::vector<uint64_t> peer_ptrs;
        at::Tensor            view;
    };

    const Buffer& acquire(const std::vector<int64_t>& shape,
                          c10::ScalarType             dtype,
                          const std::string&          tag,
                          bool*                       created = nullptr);

    void destroy();

private:
    using Key = std::tuple<std::string, std::vector<int64_t>, c10::ScalarType>;
    void*                 arena_base_;
    int64_t               reserved_, used_ = 0;
    int                   world_size_;
    std::vector<int>      peer_global_pes_;
    std::map<Key, Buffer> registry_;
    bool                  owns_arena_ = false;
    bool                  destroyed_ = false;
};

}  // namespace ulysses
