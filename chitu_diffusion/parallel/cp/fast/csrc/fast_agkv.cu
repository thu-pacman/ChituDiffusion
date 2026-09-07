#include "fast_agkv.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "ulysses_common.cuh"
#include "tma_ptx.cuh"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <nvshmemx.h>
#include <unordered_map>

namespace ulysses {
namespace {

template<int WS>
struct AgkvPeers {
    uint64_t key[WS];
    uint64_t value[WS];
};

template<int WS>
struct FlagPeers {
    uint64_t flag[WS];
};

__host__ __device__ inline int rotation_step(int ws, int dst, int src)
{
    return ((dst - src) % ws + ws) % ws;
}

__host__ __device__ inline int source_chunk_slot(int ws, int dst, int src)
{
    return ws - 1 - rotation_step(ws, dst, src);
}

template<int WS>
__global__ void fast_agkv_copy_kernel(const uint4* key,
                                      const uint4* value,
                                      AgkvPeers<WS> peers,
                                      int64_t local_vectors,
                                      int64_t local_batch_vectors,
                                      int64_t global_batch_vectors,
                                      int rank)
{
    const int64_t total = 2 * local_vectors;
    for (int64_t work = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         work < total;
         work += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const bool is_value = work >= local_vectors;
        const int64_t local = is_value ? work - local_vectors : work;
        const int64_t batch = local / local_batch_vectors;
        const int64_t within = local - batch * local_batch_vectors;
        const int64_t remote =
            batch * global_batch_vectors + static_cast<int64_t>(rank) * local_batch_vectors + within;
        const uint4 item = is_value ? value[local] : key[local];
#pragma unroll
        for (int peer = 0; peer < WS; ++peer) {
            auto* destination = reinterpret_cast<uint4*>(
                is_value ? peers.value[peer] : peers.key[peer]);
            destination[remote] = item;
        }
    }
    __threadfence_system();
}

template<int WS>
void launch_fast_agkv_ws(const void*                  key,
                         const void*                  value,
                         const std::vector<uint64_t>& key_peers,
                         const std::vector<uint64_t>& value_peers,
                         int64_t                      local_vectors,
                         int64_t                      local_batch_vectors,
                         int64_t                      global_batch_vectors,
                         int                          rank,
                         cudaStream_t                 stream)
{
    AgkvPeers<WS> peers;
    for (int peer = 0; peer < WS; ++peer) {
        peers.key[peer] = key_peers[peer];
        peers.value[peer] = value_peers[peer];
    }
    constexpr int threads = 512;
    const int64_t needed = (2 * local_vectors + threads - 1) / threads;
    const int blocks = static_cast<int>(
        std::max<int64_t>(1, std::min<int64_t>(needed, 8LL * sm_count_cached())));
    fast_agkv_copy_kernel<WS><<<blocks, threads, 0, stream>>>(
        static_cast<const uint4*>(key),
        static_cast<const uint4*>(value),
        peers,
        local_vectors,
        local_batch_vectors,
        global_batch_vectors,
        rank);
}

__global__ void u2fm2_remote_kv_copy_kernel(
    const uint4* key,
    const uint4* value,
    AgkvPeers<4> peers,
    int64_t batch,
    int64_t sequence,
    int64_t segment_stride,
    int64_t global_row_vectors,
    int64_t local_row_vectors,
    int rank)
{
    const int64_t payload_vectors = batch * sequence * local_row_vectors;
    const int64_t tensor_vectors = 2 * payload_vectors;
    const int64_t total = 2 * tensor_vectors;
    for (int64_t work = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         work < total;
         work += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const bool is_value = work >= tensor_vectors;
        const int64_t tensor_work = is_value ? work - tensor_vectors : work;
        const int head_shard = static_cast<int>(tensor_work / payload_vectors);
        const int64_t payload_work = tensor_work - head_shard * payload_vectors;
        const int64_t row = payload_work / local_row_vectors;
        const int64_t vector_in_row = payload_work - row * local_row_vectors;
        const int64_t batch_idx = row / sequence;
        const int64_t token = row - batch_idx * sequence;
        const int64_t source =
            row * global_row_vectors
            + static_cast<int64_t>(head_shard) * local_row_vectors
            + vector_in_row;
        const uint4 item = is_value ? value[source] : key[source];
#pragma unroll
        for (int full_mesh_rank = 0; full_mesh_rank < 2; ++full_mesh_rank) {
            const int dst = 2 * full_mesh_rank + head_shard;
            if (dst != rank) {
                const int slot = source_chunk_slot(4, dst, rank);
                const int64_t destination =
                    (batch_idx * 4 * segment_stride
                     + static_cast<int64_t>(slot) * segment_stride
                     + token) * local_row_vectors
                    + vector_in_row;
                auto* output = reinterpret_cast<uint4*>(
                    is_value ? peers.value[dst] : peers.key[dst]);
                output[destination] = item;
            }
        }
    }
    __threadfence_system();
}

__global__ void u2fm2_publish_remote_flags_kernel(
    FlagPeers<4> peers,
    int rank,
    int32_t epoch)
{
    const int remote_index = threadIdx.x;
    if (remote_index >= 3)
        return;
    const int dst = remote_index >= rank ? remote_index + 1 : remote_index;
    const int slot = source_chunk_slot(4, dst, rank);
    auto* remote = reinterpret_cast<int32_t*>(peers.flag[dst]) + slot;
    asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(remote), "r"(epoch) : "memory");
}

}  // namespace

void launch_fast_agkv(const void*                  key,
                      const void*                  value,
                      const std::vector<uint64_t>& key_peers,
                      const std::vector<uint64_t>& value_peers,
                      int64_t                      batch,
                      int64_t                      sequence_local,
                      int64_t                      heads,
                      int64_t                      head_dim,
                      int64_t                      elem_size,
                      int                          rank,
                      cudaStream_t                 stream)
{
    const int ws = static_cast<int>(key_peers.size());
    const int64_t row_bytes = heads * head_dim * elem_size;
    const int64_t local_batch_bytes = sequence_local * row_bytes;
    TORCH_CHECK(local_batch_bytes % 16 == 0, "Fast AGKV local batch bytes must be 16-byte aligned");
    const int64_t local_batch_vectors = local_batch_bytes / 16;
    const int64_t local_vectors = batch * local_batch_vectors;
    const int64_t global_batch_vectors = ws * local_batch_vectors;
    switch (ws) {
        case 1:
            launch_fast_agkv_ws<1>(key,
                                   value,
                                   key_peers,
                                   value_peers,
                                   local_vectors,
                                   local_batch_vectors,
                                   global_batch_vectors,
                                   rank,
                                   stream);
            break;
        case 2:
            launch_fast_agkv_ws<2>(key,
                                   value,
                                   key_peers,
                                   value_peers,
                                   local_vectors,
                                   local_batch_vectors,
                                   global_batch_vectors,
                                   rank,
                                   stream);
            break;
        case 3:
            launch_fast_agkv_ws<3>(key,
                                   value,
                                   key_peers,
                                   value_peers,
                                   local_vectors,
                                   local_batch_vectors,
                                   global_batch_vectors,
                                   rank,
                                   stream);
            break;
        case 4:
            launch_fast_agkv_ws<4>(key,
                                   value,
                                   key_peers,
                                   value_peers,
                                   local_vectors,
                                   local_batch_vectors,
                                   global_batch_vectors,
                                   rank,
                                   stream);
            break;
        case 5:
            launch_fast_agkv_ws<5>(key,
                                   value,
                                   key_peers,
                                   value_peers,
                                   local_vectors,
                                   local_batch_vectors,
                                   global_batch_vectors,
                                   rank,
                                   stream);
            break;
        case 6:
            launch_fast_agkv_ws<6>(key,
                                   value,
                                   key_peers,
                                   value_peers,
                                   local_vectors,
                                   local_batch_vectors,
                                   global_batch_vectors,
                                   rank,
                                   stream);
            break;
        case 7:
            launch_fast_agkv_ws<7>(key,
                                   value,
                                   key_peers,
                                   value_peers,
                                   local_vectors,
                                   local_batch_vectors,
                                   global_batch_vectors,
                                   rank,
                                   stream);
            break;
        case 8:
            launch_fast_agkv_ws<8>(key,
                                   value,
                                   key_peers,
                                   value_peers,
                                   local_vectors,
                                   local_batch_vectors,
                                   global_batch_vectors,
                                   rank,
                                   stream);
            break;
        default:
            TORCH_CHECK(false, "Fast AGKV world size must be in [1, 8]");
    }
}

namespace {

// One rank's K/V shard as the source of a copy-engine transfer: either the
// caller's packed input or a slot of the gathered buffer, which the layered
// schedule reads back when it relays a shard it pulled across the socket.
struct ShardSource {
    const uint8_t* key;
    const uint8_t* value;
    int64_t        batch_stride;
};

struct AgkvCEGeometry {
    int64_t batch;
    int64_t local_batch_bytes;
    int64_t global_batch_bytes;
};

// Publish one shard into `slot` of a destination's gathered buffer.
void issue_shard(const ShardSource&    source,
                 uint64_t              key_peer,
                 uint64_t              value_peer,
                 int                   slot,
                 const AgkvCEGeometry& geometry,
                 cudaStream_t          stream)
{
    for (int64_t item = 0; item < geometry.batch; ++item) {
        const int64_t source_offset      = item * source.batch_stride;
        const int64_t destination_offset = item * geometry.global_batch_bytes
                                           + static_cast<int64_t>(slot) * geometry.local_batch_bytes;
        ULYSSES_CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<uint8_t*>(key_peer) + destination_offset,
                                           source.key + source_offset,
                                           geometry.local_batch_bytes,
                                           cudaMemcpyDefault,
                                           stream));
        ULYSSES_CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<uint8_t*>(value_peer) + destination_offset,
                                           source.value + source_offset,
                                           geometry.local_batch_bytes,
                                           cudaMemcpyDefault,
                                           stream));
    }
}

}  // namespace

void launch_fast_agkv_ce(const void*                  key,
                         const void*                  value,
                         const std::vector<uint64_t>& key_peers,
                         const std::vector<uint64_t>& value_peers,
                         int64_t                      batch,
                         int64_t                      sequence_local,
                         int64_t                      heads,
                         int64_t                      head_dim,
                         int64_t                      elem_size,
                         int                          rank,
                         const CESchedule&            schedule,
                         const CENumaLayout&          layout,
                         const CEResources&           ce,
                         const std::function<void()>& phase_barrier,
                         cudaStream_t                 stream)
{
    const int            ws = static_cast<int>(key_peers.size());
    const AgkvCEGeometry geometry{batch,
                                  sequence_local * heads * head_dim * elem_size,
                                  ws * sequence_local * heads * head_dim * elem_size};
    const ShardSource    own{static_cast<const uint8_t*>(key),
                             static_cast<const uint8_t*>(value),
                             geometry.local_batch_bytes};
    // ce.streams holds one stream per rank; the schedule uses a prefix of them
    // for remote destinations and the next one for the local copy.
    const int remote_streams = std::max(1, std::min(schedule.remote_streams, std::max(ws - 1, 1)));
    const int local_slot     = ws > 1 ? remote_streams : 0;
    const int used_slots     = local_slot + 1;

    if (!schedule.layered) {
        fork_ce_streams(ce, used_slots, stream);
        int remote = 0;
        for (int step = 0; step < ws; ++step) {
            const int peer = ce_peer_at(rank, step, ws);
            const int slot = peer == rank ? local_slot : remote++ % remote_streams;
            issue_shard(own, key_peers[peer], value_peers[peer], rank, geometry, ce.streams[slot]);
        }
        join_ce_streams(ce, used_slots, stream);
        return;
    }

    // Layered: cross the socket once, then replicate inside the NUMA node.
    const int partner = layout.partner;
    fork_ce_streams(ce, used_slots, stream);
    issue_shard(own, key_peers[rank], value_peers[rank], rank, geometry, ce.streams[local_slot]);
    issue_shard(own, key_peers[partner], value_peers[partner], rank, geometry, ce.streams[0]);
    join_ce_streams(ce, used_slots, stream);
    // The relay below reads the slot the partner just wrote, so every rank has
    // to see the exchange complete before it forwards.
    phase_barrier();

    const ShardSource relayed{
        reinterpret_cast<const uint8_t*>(key_peers[rank]) + partner * geometry.local_batch_bytes,
        reinterpret_cast<const uint8_t*>(value_peers[rank]) + partner * geometry.local_batch_bytes,
        geometry.global_batch_bytes};
    fork_ce_streams(ce, used_slots, stream);
    const int    width = static_cast<int>(layout.block.size());
    const size_t index = std::find(layout.block.begin(), layout.block.end(), rank) - layout.block.begin();
    int          remote = 0;
    for (int step = 1; step < width; ++step) {
        // Rotating inside the block keeps every relay on this rank's node and
        // makes each step a permutation of it, so no receiver is a hot spot.
        const int peer = layout.block[(index + static_cast<size_t>(step)) % layout.block.size()];
        issue_shard(own,
                    key_peers[peer],
                    value_peers[peer],
                    rank,
                    geometry,
                    ce.streams[remote++ % remote_streams]);
        issue_shard(relayed,
                    key_peers[peer],
                    value_peers[peer],
                    partner,
                    geometry,
                    ce.streams[remote++ % remote_streams]);
    }
    join_ce_streams(ce, used_slots, stream);
}

std::tuple<at::Tensor, at::Tensor> all_gather_kv_4d(
    const c10::intrusive_ptr<UlyssesGroup>& group,
    at::Tensor                              key,
    at::Tensor                              value,
    std::string                             key_tag,
    std::string                             value_tag,
    bool                                    use_ce)
{
    key = key.contiguous();
    value = value.contiguous();
    TORCH_CHECK(key.is_cuda() && key.dim() == 4, "key must be a 4D CUDA tensor");
    TORCH_CHECK(value.is_cuda() && value.dim() == 4, "value must be a 4D CUDA tensor");
    TORCH_CHECK(key.sizes() == value.sizes(), "key and value shapes must match");
    TORCH_CHECK(key.scalar_type() == value.scalar_type(), "key and value dtypes must match");
    TORCH_CHECK(key.device() == value.device(), "key and value devices must match");
    TORCH_CHECK(
        key.scalar_type() == at::kHalf || key.scalar_type() == at::kBFloat16,
        "Fast AGKV supports float16 or bfloat16");
    TORCH_CHECK(key_tag != value_tag, "Fast AGKV K/V tags must be distinct");
    const int ws = static_cast<int>(group->world_size());
    TORCH_CHECK(ws >= 1 && ws <= 8, "world_size must be in [1, 8], got ", ws);
    const int64_t batch = key.size(0);
    const int64_t sequence_local = key.size(1);
    const int64_t heads = key.size(2);
    const int64_t head_dim = key.size(3);
    const int64_t elem_size = key.element_size();
    TORCH_CHECK(
        (heads * head_dim * elem_size) % 16 == 0,
        "Fast AGKV row bytes must be 16-byte aligned");
    const std::vector<int64_t> output_shape = {
        batch,
        sequence_local * ws,
        heads,
        head_dim,
    };

    const at::cuda::CUDAGuard guard(key.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const auto& key_buffer =
        group->pool().acquire(output_shape, key.scalar_type(), key_tag);
    const auto& value_buffer =
        group->pool().acquire(output_shape, value.scalar_type(), value_tag);
    if (use_ce) {
        const int           rank          = static_cast<int>(group->rank());
        const CENumaLayout& layout        = group->ce_numa_layout(key_buffer.peer_ptrs);
        const auto          phase_barrier = [&group, stream] { group->fast_barrier(stream); };
        auto                launch        = [&](const CESchedule& schedule) {
            launch_fast_agkv_ce(key.data_ptr(),
                                value.data_ptr(),
                                key_buffer.peer_ptrs,
                                value_buffer.peer_ptrs,
                                batch,
                                sequence_local,
                                heads,
                                head_dim,
                                elem_size,
                                rank,
                                schedule,
                                layout,
                                group->ce_resources(),
                                phase_barrier,
                                stream);
        };
        const CEScheduleKey schedule_key{
            static_cast<int64_t>(CEPath::all_gather), batch, sequence_local, heads, head_dim, elem_size};
        const CESchedule&   schedule = group->resolve_ce_schedule(schedule_key, [&] {
            return tune_ce_schedule(*group, layout.layered, launch, phase_barrier, stream);
        });
        launch(schedule);
    }
    else {
        launch_fast_agkv(
            key.data_ptr(),
            value.data_ptr(),
            key_buffer.peer_ptrs,
            value_buffer.peer_ptrs,
            batch,
            sequence_local,
            heads,
            head_dim,
            elem_size,
            static_cast<int>(group->rank()),
            stream);
        ULYSSES_CUDA_CHECK(cudaGetLastError());
        nvshmemx_quiet_on_stream(stream);
    }
    group->fast_barrier(stream);
    return {key_buffer.view, value_buffer.view};
}

static std::string consumed_tag(const std::string& flag_tag)
{
    return flag_tag + "__consumed";
}

static std::string ring_arrival_tag(const std::string& state_tag)
{
    return state_tag + "__arrival";
}

static std::string ring_ack_tag(const std::string& state_tag)
{
    return state_tag + "__ack";
}

static int32_t ring_ticket(int ws, int64_t epoch, int64_t phase)
{
    TORCH_CHECK(ws == 2 || ws == 4 || ws == 8, "Flash Ring CE world_size must be 2, 4, or 8");
    TORCH_CHECK(epoch > 0, "Flash Ring CE epoch must be positive");
    TORCH_CHECK(
        phase > 0 && phase < ws,
        "Flash Ring CE phase must be in [1, world_size)");
    TORCH_CHECK(
        epoch <= (INT32_MAX - phase) / (ws - 1) + 1,
        "Flash Ring CE ticket exceeds int32");
    const int64_t ticket = (epoch - 1) * (ws - 1) + phase;
    return static_cast<int32_t>(ticket);
}

// Ring phases run on the host critical path, where cudaEventCreateWithFlags
// plus cudaEventDestroy cost more than the copy they order. Recording into a
// cached event is safe because cudaStreamWaitEvent captures the event state at
// call time, so a later re-record cannot disturb an already issued wait.
enum RingEventSlot {
    kRingEventInputReady = 0,
    kRingEventSourceReady,
    kRingEventConsumed,
    kRingEventCount,
};

static cudaEvent_t ring_cached_event(int device, RingEventSlot slot)
{
    static thread_local std::unordered_map<
        int,
        std::array<cudaEvent_t, kRingEventCount>>
        cache;
    auto entry = cache.find(device);
    if (entry == cache.end()) {
        std::array<cudaEvent_t, kRingEventCount> events{};
        for (auto& event : events)
            ULYSSES_CUDA_CHECK(
                cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        entry = cache.emplace(device, events).first;
    }
    return entry->second[static_cast<int>(slot)];
}

static cudaEvent_t ring_record_event(
    cudaStream_t  stream,
    int           device,
    RingEventSlot slot)
{
    cudaEvent_t event = ring_cached_event(device, slot);
    ULYSSES_CUDA_CHECK(cudaEventRecord(event, stream));
    return event;
}

static cudaEvent_t ring_forward_done_event(int device, int phase)
{
    TORCH_CHECK(
        phase > 0 && phase < 8,
        "Flash Ring forward phase must be in [1, 8)");
    static thread_local std::unordered_map<int, std::array<cudaEvent_t, 8>> cache;
    auto entry = cache.find(device);
    if (entry == cache.end()) {
        std::array<cudaEvent_t, 8> events{};
        for (auto& event : events)
            ULYSSES_CUDA_CHECK(
                cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        entry = cache.emplace(device, events).first;
    }
    return entry->second[phase];
}

static cudaStream_t ring_ack_stream(int device)
{
    static thread_local std::unordered_map<int, cudaStream_t> cache;
    auto entry = cache.find(device);
    if (entry == cache.end()) {
        cudaStream_t stream = nullptr;
        ULYSSES_CUDA_CHECK(
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        entry = cache.emplace(device, stream).first;
    }
    return entry->second;
}

static void gate_on_consumed(
    cudaStream_t   stream,
    const int32_t* acks_local,
    int            peer,
    int64_t        epoch)
{
    const CUresult status = cuStreamWaitValue32(
        stream,
        reinterpret_cast<CUdeviceptr>(acks_local + peer),
        static_cast<cuuint32_t>(epoch - 1),
        CU_STREAM_WAIT_VALUE_GEQ);
    TORCH_CHECK(
        status == CUDA_SUCCESS,
        "cuStreamWaitValue32 failed (",
        static_cast<int>(status),
        "); Fast AGKV transports need stream memory operations");
}

static void stream_write_value32(
    cudaStream_t stream,
    int32_t*     address,
    int32_t      value)
{
    const CUresult status = cuStreamWriteValue32(
        stream,
        reinterpret_cast<CUdeviceptr>(address),
        static_cast<cuuint32_t>(value),
        CU_STREAM_WRITE_VALUE_DEFAULT);
    TORCH_CHECK(
        status == CUDA_SUCCESS,
        "cuStreamWriteValue32 failed (",
        static_cast<int>(status),
        "); Fast AGKV transports need stream memory operations");
}

struct LogicalBarrierPeers {
    uint64_t pointers[8];
};

__global__ void logical_subgroup_barrier_kernel(
    uint64_t* local,
    LogicalBarrierPeers peers,
    int peer_count,
    int logical_rank,
    uint64_t epoch)
{
    const int peer = threadIdx.x;
    if (peer >= peer_count)
        return;
    auto* remote = reinterpret_cast<uint64_t*>(peers.pointers[peer]) + logical_rank;
    st_release_sys_u64(remote, epoch);
    uint64_t observed;
    do {
        observed = ld_acquire_sys_u64(local + peer);
    } while (observed < epoch);
}

static int validate_logical_peers(
    const c10::intrusive_ptr<UlyssesGroup>& group,
    const std::vector<int64_t>&             peer_ranks)
{
    TORCH_CHECK(!peer_ranks.empty() && peer_ranks.size() <= 8, "logical subgroup size must be in [1, 8]");
    std::vector<int> seen(group->world_size(), 0);
    int logical_rank = -1;
    for (size_t i = 0; i < peer_ranks.size(); ++i) {
        const int64_t peer = peer_ranks[i];
        TORCH_CHECK(peer >= 0 && peer < group->world_size(), "logical peer is out of WORLD range");
        TORCH_CHECK(!seen[peer], "logical peer list contains duplicates");
        seen[peer] = 1;
        if (peer == group->rank())
            logical_rank = static_cast<int>(i);
    }
    TORCH_CHECK(logical_rank >= 0, "logical peer list does not contain this WORLD rank");
    return logical_rank;
}

static void launch_logical_subgroup_barrier(
    const SymmetricHeapPool::Buffer& barrier,
    const std::vector<int64_t>&      peer_ranks,
    int                              logical_rank,
    int64_t                          epoch,
    cudaStream_t                     stream)
{
    LogicalBarrierPeers peers{};
    for (size_t i = 0; i < peer_ranks.size(); ++i)
        peers.pointers[i] = barrier.peer_ptrs[peer_ranks[i]];
    logical_subgroup_barrier_kernel<<<1, 32, 0, stream>>>(
        static_cast<uint64_t*>(barrier.sym_base),
        peers,
        static_cast<int>(peer_ranks.size()),
        logical_rank,
        static_cast<uint64_t>(epoch));
    ULYSSES_CUDA_CHECK(cudaGetLastError());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, std::vector<int64_t>>
ring_ce_prepare_kv_4d(
    const c10::intrusive_ptr<UlyssesGroup>& group,
    at::Tensor                              key,
    at::Tensor                              value,
    std::string                             key_tag,
    std::string                             value_tag,
    std::string                             state_tag,
    int64_t                                 epoch)
{
    TORCH_CHECK(key.is_cuda() && key.dim() == 4, "Flash Ring key must be a 4D CUDA tensor");
    TORCH_CHECK(key.is_contiguous(), "Flash Ring key must be contiguous for zero-copy local use");
    TORCH_CHECK(value.is_cuda() && value.sizes() == key.sizes(), "Flash Ring value shape must match key");
    TORCH_CHECK(value.is_contiguous(), "Flash Ring value must be contiguous for zero-copy local use");
    TORCH_CHECK(value.device() == key.device(), "Flash Ring K/V devices must match");
    TORCH_CHECK(value.scalar_type() == key.scalar_type(), "Flash Ring K/V dtypes must match");
    TORCH_CHECK(
        key.scalar_type() == at::kHalf || key.scalar_type() == at::kBFloat16,
        "Flash Ring CE supports float16 or bfloat16");
    TORCH_CHECK(
        key_tag != value_tag && key_tag != state_tag && value_tag != state_tag,
        "Flash Ring CE K/V/state tags must be distinct");

    const int ws = static_cast<int>(group->world_size());
    const int rank = static_cast<int>(group->rank());
    const int right = (rank + 1) % ws;
    const int32_t ticket = ring_ticket(ws, epoch, 1);
    const int slot = ticket & 1;
    std::vector<int64_t> landing_shape = key.sizes().vec();
    landing_shape.insert(landing_shape.begin(), 2);

    bool arrivals_created = false;
    bool acks_created = false;
    const auto& key_buffer =
        group->pool().acquire(landing_shape, key.scalar_type(), key_tag);
    const auto& value_buffer =
        group->pool().acquire(landing_shape, value.scalar_type(), value_tag);
    const auto& arrivals = group->pool().acquire(
        {2}, at::kInt, ring_arrival_tag(state_tag), &arrivals_created);
    const auto& acks = group->pool().acquire(
        {2}, at::kInt, ring_ack_tag(state_tag), &acks_created);
    const at::cuda::CUDAGuard guard(key.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (arrivals_created || acks_created || epoch == 1) {
        ULYSSES_CUDA_CHECK(cudaMemsetAsync(
            arrivals.view.data_ptr(), 0, arrivals.view.nbytes(), stream));
        ULYSSES_CUDA_CHECK(cudaMemsetAsync(
            acks.view.data_ptr(), 0, acks.view.nbytes(), stream));
        group->fast_barrier(stream);
    }

    const int device = key.device().index();
    cudaEvent_t input_ready =
        ring_record_event(stream, device, kRingEventInputReady);
    cudaStream_t copy_stream = group->ce_resources().streams[right];
    ULYSSES_CUDA_CHECK(cudaStreamWaitEvent(copy_stream, input_ready, 0));
    if (ticket > 2)
        gate_on_consumed(
            // gate_on_consumed waits for its final argument minus one.
            copy_stream, acks.view.data_ptr<int32_t>(), slot, ticket - 1);

    const size_t payload_bytes = static_cast<size_t>(key.nbytes());
    ULYSSES_CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<uint8_t*>(key_buffer.peer_ptrs[right])
            + static_cast<size_t>(slot) * payload_bytes,
        key.data_ptr(),
        payload_bytes,
        cudaMemcpyDefault,
        copy_stream));
    ULYSSES_CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<uint8_t*>(value_buffer.peer_ptrs[right])
            + static_cast<size_t>(slot) * payload_bytes,
        value.data_ptr(),
        payload_bytes,
        cudaMemcpyDefault,
        copy_stream));
    // Stream memory writes are ordered after both CE copies. The right peer
    // therefore cannot observe this ticket before the complete K/V payload.
    stream_write_value32(
        copy_stream,
        reinterpret_cast<int32_t*>(arrivals.peer_ptrs[right]) + slot,
        ticket);

    // Prepost the complete forwarding chain while phase-zero attention runs.
    // GPU-side ticket waits release each hop as soon as its source lands; no
    // host acquire/forward round trip remains between attention phases.
    for (int64_t phase = 1; phase < ws - 1; ++phase) {
        const int32_t current_ticket = ring_ticket(ws, epoch, phase);
        const int32_t next_ticket = ring_ticket(ws, epoch, phase + 1);
        const int source_slot = current_ticket & 1;
        const int target_slot = next_ticket & 1;
        gate_on_consumed(
            copy_stream,
            arrivals.view.data_ptr<int32_t>(),
            source_slot,
            static_cast<int64_t>(current_ticket) + 1);
        gate_on_consumed(
            copy_stream,
            acks.view.data_ptr<int32_t>(),
            target_slot,
            next_ticket - 1);
        ULYSSES_CUDA_CHECK(cudaMemcpyAsync(
            reinterpret_cast<uint8_t*>(key_buffer.peer_ptrs[right])
                + static_cast<size_t>(target_slot) * payload_bytes,
            reinterpret_cast<const uint8_t*>(key_buffer.view.data_ptr())
                + static_cast<size_t>(source_slot) * payload_bytes,
            payload_bytes,
            cudaMemcpyDefault,
            copy_stream));
        ULYSSES_CUDA_CHECK(cudaMemcpyAsync(
            reinterpret_cast<uint8_t*>(value_buffer.peer_ptrs[right])
                + static_cast<size_t>(target_slot) * payload_bytes,
            reinterpret_cast<const uint8_t*>(value_buffer.view.data_ptr())
                + static_cast<size_t>(source_slot) * payload_bytes,
            payload_bytes,
            cudaMemcpyDefault,
            copy_stream));
        stream_write_value32(
            copy_stream,
            reinterpret_cast<int32_t*>(arrivals.peer_ptrs[right]) + target_slot,
            next_ticket);
        ULYSSES_CUDA_CHECK(cudaEventRecord(
            ring_forward_done_event(device, static_cast<int>(phase)),
            copy_stream));
    }
    // Every later phase reuses these same addresses. Resolving them once here
    // keeps the per-phase calls free of tag strings and pool lookups, which
    // otherwise dominate the host critical path at short sequences.
    const int left = (rank + ws - 1) % ws;
    const std::vector<int64_t> handles = {
        static_cast<int64_t>(key_buffer.peer_ptrs[right]),
        static_cast<int64_t>(value_buffer.peer_ptrs[right]),
        static_cast<int64_t>(arrivals.peer_ptrs[right]),
        reinterpret_cast<int64_t>(acks.view.data_ptr<int32_t>()),
        static_cast<int64_t>(acks.peer_ptrs[left]),
    };
    return {
        key_buffer.view,
        value_buffer.view,
        arrivals.view,
        acks.view,
        handles,
    };
}

void ring_ce_forward_kv_4d(
    const c10::intrusive_ptr<UlyssesGroup>& group,
    at::Tensor                              landing_key,
    at::Tensor                              landing_value,
    int64_t                                 peer_key_ptr,
    int64_t                                 peer_value_ptr,
    int64_t                                 peer_arrival_ptr,
    int64_t                                 local_ack_ptr,
    int64_t                                 epoch,
    int64_t                                 phase)
{
    const int ws = static_cast<int>(group->world_size());
    TORCH_CHECK(phase > 0 && phase < ws - 1, "Flash Ring CE has no forward after the final phase");
    const int32_t current_ticket = ring_ticket(ws, epoch, phase);
    const int32_t next_ticket = ring_ticket(ws, epoch, phase + 1);
    TORCH_CHECK(
        landing_key.is_cuda() && landing_key.dim() == 5 && landing_key.size(0) == 2
            && landing_key.is_contiguous(),
        "Flash Ring landing key must be a contiguous [2,B,S,H,D] CUDA tensor");
    TORCH_CHECK(
        landing_value.sizes() == landing_key.sizes()
            && landing_value.scalar_type() == landing_key.scalar_type()
            && landing_value.is_contiguous(),
        "Flash Ring landing K/V tensors must match");

    const int rank = static_cast<int>(group->rank());
    const int right = (rank + 1) % ws;
    const int source_slot = current_ticket & 1;
    const int target_slot = next_ticket & 1;

    const at::cuda::CUDAGuard guard(landing_key.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaEvent_t source_ready = ring_record_event(
        stream, landing_key.device().index(), kRingEventSourceReady);
    cudaStream_t copy_stream = group->ce_resources().streams[right];
    ULYSSES_CUDA_CHECK(cudaStreamWaitEvent(copy_stream, source_ready, 0));
    gate_on_consumed(
        // For the first use of slot zero this waits for ticket zero; later
        // uses wait for next_ticket - 2, the prior owner of the same slot.
        copy_stream,
        reinterpret_cast<int32_t*>(local_ack_ptr),
        target_slot,
        next_ticket - 1);

    // Both slots share one allocation, so a slot spans half the landing bytes.
    const size_t payload_bytes = static_cast<size_t>(landing_key.nbytes()) / 2;
    ULYSSES_CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<uint8_t*>(peer_key_ptr)
            + static_cast<size_t>(target_slot) * payload_bytes,
        static_cast<const uint8_t*>(landing_key.data_ptr())
            + static_cast<size_t>(source_slot) * payload_bytes,
        payload_bytes,
        cudaMemcpyDefault,
        copy_stream));
    ULYSSES_CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<uint8_t*>(peer_value_ptr)
            + static_cast<size_t>(target_slot) * payload_bytes,
        static_cast<const uint8_t*>(landing_value.data_ptr())
            + static_cast<size_t>(source_slot) * payload_bytes,
        payload_bytes,
        cudaMemcpyDefault,
        copy_stream));
    stream_write_value32(
        copy_stream,
        reinterpret_cast<int32_t*>(peer_arrival_ptr) + target_slot,
        next_ticket);
}

at::Tensor ring_ce_wait_ready(
    at::Tensor arrivals,
    int64_t   world_size,
    int64_t   epoch,
    int64_t   phase)
{
    const int32_t ticket =
        ring_ticket(static_cast<int>(world_size), epoch, phase);
    TORCH_CHECK(
        arrivals.is_cuda() && arrivals.scalar_type() == at::kInt
            && arrivals.numel() == 2,
        "Flash Ring arrivals must be a two-element CUDA int32 tensor");
    const at::cuda::CUDAGuard guard(arrivals.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const CUresult status = cuStreamWaitValue32(
        stream,
        reinterpret_cast<CUdeviceptr>(
            arrivals.data_ptr<int32_t>() + (ticket & 1)),
        static_cast<cuuint32_t>(ticket),
        CU_STREAM_WAIT_VALUE_GEQ);
    TORCH_CHECK(
        status == CUDA_SUCCESS,
        "cuStreamWaitValue32 failed (",
        static_cast<int>(status),
        "); Flash Ring CE needs stream memory operations");
    return arrivals;
}

void ring_ce_publish_consumed(
    const c10::intrusive_ptr<UlyssesGroup>& group,
    int64_t                                 peer_ack_ptr,
    int64_t                                 device_index,
    int64_t                                 epoch,
    int64_t                                 phase)
{
    const int ws = static_cast<int>(group->world_size());
    const int32_t ticket = ring_ticket(ws, epoch, phase);
    const int slot = ticket & 1;
    const at::cuda::CUDAGuard guard(
        at::Device(at::kCUDA, static_cast<c10::DeviceIndex>(device_index)));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaEvent_t consumed = ring_record_event(
        stream, static_cast<int>(device_index), kRingEventConsumed);
    cudaStream_t ack_stream = ring_ack_stream(static_cast<int>(device_index));
    ULYSSES_CUDA_CHECK(cudaStreamWaitEvent(ack_stream, consumed, 0));
    if (phase < ws - 1)
        ULYSSES_CUDA_CHECK(cudaStreamWaitEvent(
            ack_stream,
            ring_forward_done_event(
                static_cast<int>(device_index), static_cast<int>(phase)),
            0));
    // ACKs must not sit behind a preposted future-arrival wait. The dedicated
    // stream waits for both readers of this slot (attention and forwarding),
    // then releases the left peer independently of the forwarding chain.
    stream_write_value32(
        ack_stream,
        reinterpret_cast<int32_t*>(peer_ack_ptr) + slot,
        ticket);
}

at::Tensor logical_subgroup_all_to_all_single_4d_ce(
    const c10::intrusive_ptr<UlyssesGroup>& group,
    at::Tensor                              input,
    std::vector<int64_t>                    peer_ranks,
    int64_t                                 mode,
    std::string                             tag,
    int64_t                                 storage_slots,
    int64_t                                 storage_index)
{
    input = input.contiguous();
    const int logical_ws = static_cast<int>(peer_ranks.size());
    const int logical_rank = validate_logical_peers(group, peer_ranks);
    TORCH_CHECK(storage_slots > 0 && storage_index >= 0 && storage_index < storage_slots,
                "invalid symmetric storage slot");
    TORCH_CHECK(input.is_cuda() && input.dim() == 4, "input must be a 4D CUDA tensor");
    TORCH_CHECK(input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
                "dtype must be float16 or bfloat16");
    TORCH_CHECK(mode == 0 || mode == 1, "mode must be 0 or 1");

    Ulysses4DDims dims;
    dims.b = static_cast<int>(input.size(0));
    dims.d = static_cast<int>(input.size(3));
    dims.rank = logical_rank;
    std::vector<int64_t> out_shape;
    if (mode == 0) {
        TORCH_CHECK(input.size(2) % logical_ws == 0, "heads must divide the logical subgroup size");
        dims.s_local = static_cast<int>(input.size(1));
        dims.n_global = static_cast<int>(input.size(2));
        dims.s_global = dims.s_local * logical_ws;
        dims.n_local = dims.n_global / logical_ws;
        out_shape = {dims.b, dims.s_global, dims.n_local, dims.d};
    } else {
        TORCH_CHECK(input.size(1) % logical_ws == 0, "sequence must divide the logical subgroup size");
        dims.s_global = static_cast<int>(input.size(1));
        dims.n_local = static_cast<int>(input.size(2));
        dims.s_local = dims.s_global / logical_ws;
        dims.n_global = dims.n_local * logical_ws;
        out_shape = {dims.b, dims.s_local, dims.n_global, dims.d};
    }
    TORCH_CHECK(static_cast<int64_t>(dims.d) * input.element_size() % 16 == 0,
                "head_dim rows must be 16-byte aligned");

    const int64_t out_numel = input.numel();
    const auto& buffer = group->pool().acquire(
        {storage_slots, out_numel}, input.scalar_type(), tag);
    const int64_t slot_bytes = out_numel * input.element_size();
    std::vector<uint64_t> logical_ptrs(logical_ws);
    for (int i = 0; i < logical_ws; ++i)
        logical_ptrs[i] = buffer.peer_ptrs[peer_ranks[i]] + storage_index * slot_bytes;
    const at::cuda::CUDAGuard guard(input.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // Untuned, unlike the group-wide op: a logical subgroup has no collective barrier of
    // its own to time candidates against, and ranks in different subgroups do not run
    // this call in lockstep, so the group-wide reduction that keeps the tuner's barriers
    // aligned is unavailable here. Keep the stream-per-peer fan-out.
    const CESchedule schedule{std::max(logical_ws - 1, 1), false};
    launch_a2a_ce(
        input.data_ptr(), logical_ptrs, dims, static_cast<int>(mode),
        static_cast<int>(input.element_size()), schedule, group->ce_resources(), stream);
    return buffer.view[storage_index].view(out_shape);
}

at::Tensor logical_subgroup_barrier(
    const c10::intrusive_ptr<UlyssesGroup>& group,
    std::vector<int64_t>                    peer_ranks,
    std::string                             tag,
    int64_t                                 epoch)
{
    TORCH_CHECK(epoch > 0, "logical subgroup epoch must be positive");
    const int logical_rank = validate_logical_peers(group, peer_ranks);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    bool created = false;
    const auto& barrier = group->pool().acquire(
        {static_cast<int64_t>(peer_ranks.size())}, at::kLong, tag, &created);
    if (created) {
        ULYSSES_CUDA_CHECK(cudaMemsetAsync(
            barrier.view.data_ptr(), 0, barrier.view.nbytes(), stream));
        // Every WORLD rank creates one logical-U barrier on the same call. This
        // one-time initialization fence prevents a peer's epoch-1 publication
        // from racing another rank's local memset.
        group->fast_barrier(stream);
    }
    launch_logical_subgroup_barrier(
        barrier, peer_ranks, logical_rank, epoch, stream);
    return barrier.view;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
u2fm2_prepare_qkv(
    const c10::intrusive_ptr<UlyssesGroup>& group,
    at::Tensor                              query,
    at::Tensor                              key,
    at::Tensor                              value,
    std::string                             query_tag,
    std::string                             key_tag,
    std::string                             value_tag,
    std::string                             flag_tag,
    std::string                             ulysses_barrier_tag,
    int64_t                                 ulysses_epoch,
    int64_t                                 full_mesh_epoch,
    int64_t                                 segment_stride,
    int64_t                                 remote_blocks)
{
    query = query.contiguous();
    key = key.contiguous();
    value = value.contiguous();
    TORCH_CHECK(group->world_size() == 4, "U2xFM2 requires a WORLD group of size 4");
    TORCH_CHECK(query.is_cuda() && query.dim() == 4, "query must be a 4D CUDA tensor");
    TORCH_CHECK(key.sizes() == query.sizes() && value.sizes() == query.sizes(),
                "U2xFM2 requires matching Q/K/V shapes");
    TORCH_CHECK(query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16,
                "U2xFM2 supports float16 or bfloat16");
    TORCH_CHECK(key.scalar_type() == query.scalar_type() && value.scalar_type() == query.scalar_type(),
                "U2xFM2 requires matching Q/K/V dtypes");
    TORCH_CHECK(query.size(2) % 2 == 0, "U2xFM2 requires H % 2 == 0");
    TORCH_CHECK(segment_stride >= query.size(1), "segment_stride must cover the local sequence");
    TORCH_CHECK(ulysses_epoch > 0 && full_mesh_epoch > 0,
                "Ulysses and Full-Mesh epochs must be positive");
    TORCH_CHECK(remote_blocks > 0 && remote_blocks <= 64,
                "U2xFM2 remote block count must be in [1, 64]");
    TORCH_CHECK(query_tag != key_tag && query_tag != value_tag && key_tag != value_tag,
                "Q/K/V symmetric tags must be distinct");

    const int rank = static_cast<int>(group->rank());
    const int u_rank = rank % 2;
    const int f_rank = rank / 2;
    const std::vector<int64_t> u_peers = {2 * f_rank, 2 * f_rank + 1};
    const int logical_u_rank = u_rank;
    const int64_t batch = query.size(0);
    const int64_t sequence = query.size(1);
    const int64_t heads_global = query.size(2);
    const int64_t heads_local = heads_global / 2;
    const int64_t head_dim = query.size(3);
    const int64_t elem = query.element_size();
    const int64_t global_row_bytes = heads_global * head_dim * elem;
    const int64_t local_row_bytes = heads_local * head_dim * elem;
    TORCH_CHECK(local_row_bytes % 16 == 0, "U2xFM2 head shard rows must be 16-byte aligned");

    const std::vector<int64_t> q_shape = {
        batch, sequence * 2, heads_local, head_dim};
    const std::vector<int64_t> kv_shape = {
        batch, segment_stride * 4, heads_local, head_dim};
    const auto& q_buffer = group->pool().acquire(
        {1, query.numel()}, query.scalar_type(), query_tag);
    bool key_created = false;
    bool value_created = false;
    bool flags_created = false;
    bool acks_created = false;
    bool ubar_created = false;
    const auto& key_buffer = group->pool().acquire(
        kv_shape, key.scalar_type(), key_tag, &key_created);
    const auto& value_buffer = group->pool().acquire(
        kv_shape, value.scalar_type(), value_tag, &value_created);
    const auto& flag_buffer = group->pool().acquire(
        {4}, at::kInt, flag_tag, &flags_created);
    const auto& ack_buffer = group->pool().acquire(
        {4}, at::kInt, consumed_tag(flag_tag), &acks_created);
    const auto& ubar_buffer = group->pool().acquire(
        {2}, at::kLong, ulysses_barrier_tag, &ubar_created);

    const at::cuda::CUDAGuard guard(query.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (key_created || value_created || flags_created || acks_created || ubar_created) {
        if (key_created)
            ULYSSES_CUDA_CHECK(cudaMemsetAsync(
                key_buffer.view.data_ptr(), 0, key_buffer.view.nbytes(), stream));
        if (value_created)
            ULYSSES_CUDA_CHECK(cudaMemsetAsync(
                value_buffer.view.data_ptr(), 0, value_buffer.view.nbytes(), stream));
        if (flags_created)
            ULYSSES_CUDA_CHECK(cudaMemsetAsync(
                flag_buffer.view.data_ptr(), 0, flag_buffer.view.nbytes(), stream));
        if (acks_created)
            ULYSSES_CUDA_CHECK(cudaMemsetAsync(
                ack_buffer.view.data_ptr(), 0, ack_buffer.view.nbytes(), stream));
        if (ubar_created)
            ULYSSES_CUDA_CHECK(cudaMemsetAsync(
                ubar_buffer.view.data_ptr(), 0, ubar_buffer.view.nbytes(), stream));
        group->fast_barrier(stream);
    }

    at::Tensor epoch_guard = at::full(
        {1},
        static_cast<int32_t>(full_mesh_epoch),
        query.options().dtype(at::kInt));
    cudaEvent_t input_ready;
    ULYSSES_CUDA_CHECK(cudaEventCreateWithFlags(&input_ready, cudaEventDisableTiming));
    ULYSSES_CUDA_CHECK(cudaEventRecord(input_ready, stream));
    std::vector<cudaEvent_t> q_done;
    q_done.reserve(2);
    const auto& ce = group->ce_resources();
    const int remote_stream_index = (rank + 2) % 4;
    cudaStream_t remote_stream = ce.streams[remote_stream_index];
    ULYSSES_CUDA_CHECK(cudaStreamWaitEvent(remote_stream, input_ready, 0));
    AgkvPeers<4> remote_peers{};
    FlagPeers<4> remote_flags{};
    for (int peer = 0; peer < 4; ++peer) {
        remote_peers.key[peer] = key_buffer.peer_ptrs[peer];
        remote_peers.value[peer] = value_buffer.peer_ptrs[peer];
        remote_flags.flag[peer] = flag_buffer.peer_ptrs[peer];
    }
    constexpr int remote_threads = 512;
    u2fm2_remote_kv_copy_kernel<<<
        static_cast<int>(remote_blocks), remote_threads, 0, remote_stream>>>(
        static_cast<const uint4*>(key.data_ptr()),
        static_cast<const uint4*>(value.data_ptr()),
        remote_peers,
        batch,
        sequence,
        segment_stride,
        global_row_bytes / 16,
        local_row_bytes / 16,
        rank);
    u2fm2_publish_remote_flags_kernel<<<1, 32, 0, remote_stream>>>(
        remote_flags, rank, static_cast<int32_t>(full_mesh_epoch));
    ULYSSES_CUDA_CHECK(cudaGetLastError());

    for (int dst = 0; dst < 4; ++dst) {
        cudaStream_t copy_stream = ce.streams[dst];
        ULYSSES_CUDA_CHECK(cudaStreamWaitEvent(copy_stream, input_ready, 0));
        const int dst_u = dst % 2;
        const int dst_f = dst / 2;
        const int64_t source_head_offset = dst_u * local_row_bytes;

        if (dst_f == f_rank) {
            for (int64_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
                const auto* source = static_cast<const uint8_t*>(query.data_ptr())
                    + batch_idx * sequence * global_row_bytes
                    + source_head_offset;
                auto* destination = reinterpret_cast<uint8_t*>(q_buffer.peer_ptrs[dst])
                    + (batch_idx * 2 * sequence + u_rank * sequence) * local_row_bytes;
                ULYSSES_CUDA_CHECK(cudaMemcpy2DAsync(
                    destination, local_row_bytes,
                    source, global_row_bytes,
                    local_row_bytes, sequence,
                    cudaMemcpyDefault, copy_stream));
            }
            if (dst != rank) {
                cudaEvent_t done;
                ULYSSES_CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
                ULYSSES_CUDA_CHECK(cudaEventRecord(done, copy_stream));
                q_done.push_back(done);
            }
        }

        if (dst == rank) {
            const int slot = 3;
            for (int64_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
                const int64_t source_batch_offset = batch_idx * sequence * global_row_bytes;
                const int64_t destination_row =
                    (batch_idx * 4 * segment_stride + slot * segment_stride);
                auto* key_destination = reinterpret_cast<uint8_t*>(key_buffer.peer_ptrs[dst])
                    + destination_row * local_row_bytes;
                auto* value_destination = reinterpret_cast<uint8_t*>(value_buffer.peer_ptrs[dst])
                    + destination_row * local_row_bytes;
                const auto* key_source = static_cast<const uint8_t*>(key.data_ptr())
                    + source_batch_offset + source_head_offset;
                const auto* value_source = static_cast<const uint8_t*>(value.data_ptr())
                    + source_batch_offset + source_head_offset;
                ULYSSES_CUDA_CHECK(cudaMemcpy2DAsync(
                    key_destination, local_row_bytes,
                    key_source, global_row_bytes,
                    local_row_bytes, sequence,
                    cudaMemcpyDefault, copy_stream));
                ULYSSES_CUDA_CHECK(cudaMemcpy2DAsync(
                    value_destination, local_row_bytes,
                    value_source, global_row_bytes,
                    local_row_bytes, sequence,
                    cudaMemcpyDefault, copy_stream));
                if (segment_stride > sequence) {
                    const size_t padding_bytes =
                        static_cast<size_t>(segment_stride - sequence) * local_row_bytes;
                    ULYSSES_CUDA_CHECK(cudaMemsetAsync(
                        key_destination + sequence * local_row_bytes,
                        0,
                        padding_bytes,
                        copy_stream));
                    ULYSSES_CUDA_CHECK(cudaMemsetAsync(
                        value_destination + sequence * local_row_bytes,
                        0,
                        padding_bytes,
                        copy_stream));
                }
            }
            ULYSSES_CUDA_CHECK(cudaMemcpyAsync(
                reinterpret_cast<int32_t*>(flag_buffer.peer_ptrs[dst]) + slot,
                epoch_guard.data_ptr<int32_t>(),
                sizeof(int32_t),
                cudaMemcpyDefault,
                copy_stream));
            cudaEvent_t done;
            ULYSSES_CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
            ULYSSES_CUDA_CHECK(cudaEventRecord(done, copy_stream));
            q_done.push_back(done);
        }
    }

    for (cudaEvent_t done : q_done) {
        ULYSSES_CUDA_CHECK(cudaStreamWaitEvent(stream, done, 0));
        ULYSSES_CUDA_CHECK(cudaEventDestroy(done));
    }
    ULYSSES_CUDA_CHECK(cudaEventDestroy(input_ready));
    launch_logical_subgroup_barrier(
        ubar_buffer, u_peers, logical_u_rank, ulysses_epoch, stream);

    return {
        q_buffer.view[0].view(q_shape),
        key_buffer.view,
        value_buffer.view,
        flag_buffer.view,
        epoch_guard,
    };
}

}  // namespace ulysses
