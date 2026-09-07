#include "ce_schedule.cuh"
#include "ulysses_common.cuh"
#include "ulysses_group.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <iterator>
#include <map>
#include <string>

namespace ulysses {
namespace {

// NUMA node of one CUDA device, from the sysfs entry its PCI address names.
// Returns -1 when the host does not report one (single-socket machines, and
// containers without /sys/bus/pci mounted).
int numa_node_of(int device)
{
    char address[32] = {};
    if (cudaDeviceGetPCIBusId(address, static_cast<int>(sizeof(address)), device) != cudaSuccess)
        return -1;
    std::string path = "/sys/bus/pci/devices/";
    for (const char* c = address; *c != '\0'; ++c)
        path += static_cast<char>(*c >= 'A' && *c <= 'Z' ? *c - 'A' + 'a' : *c);  // sysfs is lower case
    path += "/numa_node";
    std::FILE* file = std::fopen(path.c_str(), "r");
    if (file == nullptr)
        return -1;
    int node = -1;
    if (std::fscanf(file, "%d", &node) != 1)
        node = -1;
    std::fclose(file);
    return node;
}

// Device ordinal backing a peer's symmetric allocation. Peer pointers are P2P
// mapped into this process, so the driver can name their device.
int device_of(uint64_t pointer)
{
    cudaPointerAttributes attributes{};
    if (cudaPointerGetAttributes(&attributes, reinterpret_cast<void*>(pointer)) != cudaSuccess) {
        cudaGetLastError();  // an unrecognised pointer leaves a sticky error behind
        return -1;
    }
    return attributes.device;
}

}  // namespace

CENumaLayout resolve_ce_numa_layout(int rank, const std::vector<uint64_t>& peer_ptrs)
{
    const int    ws = static_cast<int>(peer_ptrs.size());
    CENumaLayout layout;
    layout.nodes.assign(ws, -1);
    std::map<int, int> cache;  // device -> node, so a repeated device is read once
    for (int peer = 0; peer < ws; ++peer) {
        const int device = device_of(peer_ptrs[peer]);
        if (device < 0)
            continue;
        auto entry = cache.find(device);
        if (entry == cache.end())
            entry = cache.emplace(device, numa_node_of(device)).first;
        layout.nodes[peer] = entry->second;
    }

    std::map<int, std::vector<int>> blocks;
    for (int peer = 0; peer < ws; ++peer)
        if (layout.nodes[peer] >= 0)
            blocks[layout.nodes[peer]].push_back(peer);
    layout.block.assign(1, rank);
    if (blocks.size() != 2)
        return layout;
    const std::vector<int>& first  = blocks.begin()->second;
    const std::vector<int>& second = std::next(blocks.begin())->second;
    if (first.size() != second.size() || first.size() < 2)
        return layout;

    const bool               mine  = layout.nodes[rank] == blocks.begin()->first;
    const std::vector<int>&  block = mine ? first : second;
    const std::vector<int>&  other = mine ? second : first;
    const size_t             index = std::find(block.begin(), block.end(), rank) - block.begin();
    if (index >= block.size())
        return layout;
    // Pair the two blocks by position, which every rank derives identically.
    layout.block   = block;
    layout.partner = other[index];
    layout.layered = true;
    return layout;
}

void fork_ce_streams(const CEResources& ce, int slots, cudaStream_t stream)
{
    cudaEvent_t ready;
    ULYSSES_CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    ULYSSES_CUDA_CHECK(cudaEventRecord(ready, stream));
    for (int slot = 0; slot < slots; ++slot)
        ULYSSES_CUDA_CHECK(cudaStreamWaitEvent(ce.streams[slot], ready, 0));
    ULYSSES_CUDA_CHECK(cudaEventDestroy(ready));
}

void join_ce_streams(const CEResources& ce, int slots, cudaStream_t stream)
{
    for (int slot = 0; slot < slots; ++slot) {
        cudaEvent_t done;
        ULYSSES_CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
        ULYSSES_CUDA_CHECK(cudaEventRecord(done, ce.streams[slot]));
        ULYSSES_CUDA_CHECK(cudaStreamWaitEvent(stream, done, 0));
        ULYSSES_CUDA_CHECK(cudaEventDestroy(done));
    }
}

namespace {

// Replace each candidate's local time with its worst time across the group.
//
// The group MUST agree on one schedule: the layered schedules issue an extra
// phase barrier per call, so if a near tie left two ranks on different
// schedules their barrier epochs would drift apart and the group would hang.
// (The TMA/non-TMA tuner can afford a per-rank decision precisely because those
// paths have identical barrier counts.) Reducing to the worst rank also matches
// what a collective actually costs, which is set by its slowest participant.
void reduce_candidate_times(UlyssesGroup&                group,
                            std::vector<float>&          times,
                            const std::function<void()>& barrier,
                            cudaStream_t                 stream)
{
    const int      ws    = static_cast<int>(group.world_size());
    const int      rank  = static_cast<int>(group.rank());
    const int64_t  count = static_cast<int64_t>(times.size());
    const int64_t  row   = count * static_cast<int64_t>(sizeof(float));
    const auto&    board = group.pool().acquire({ws, count}, at::kFloat, "__ulysses_ce_tune__");
    uint8_t* const mine  = reinterpret_cast<uint8_t*>(board.peer_ptrs[rank]) + rank * row;
    ULYSSES_CUDA_CHECK(cudaMemcpyAsync(mine, times.data(), row, cudaMemcpyHostToDevice, stream));
    for (int step = 1; step < ws; ++step) {
        const int peer = ce_peer_at(rank, step, ws);
        ULYSSES_CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<uint8_t*>(board.peer_ptrs[peer]) + rank * row,
                                           mine,
                                           row,
                                           cudaMemcpyDefault,
                                           stream));
    }
    barrier();
    std::vector<float> gathered(static_cast<size_t>(ws) * times.size());
    ULYSSES_CUDA_CHECK(cudaMemcpyAsync(gathered.data(),
                                       reinterpret_cast<void*>(board.peer_ptrs[rank]),
                                       gathered.size() * sizeof(float),
                                       cudaMemcpyDeviceToHost,
                                       stream));
    ULYSSES_CUDA_CHECK(cudaStreamSynchronize(stream));
    for (size_t candidate = 0; candidate < times.size(); ++candidate)
        for (int peer = 0; peer < ws; ++peer)
            times[candidate] = std::max(times[candidate], gathered[peer * times.size() + candidate]);
}

}  // namespace

CESchedule tune_ce_schedule(UlyssesGroup&                                 group,
                            bool                                          layered_possible,
                            const std::function<void(const CESchedule&)>& launch,
                            const std::function<void()>&                  barrier,
                            cudaStream_t                                  stream)
{
    const int ws = static_cast<int>(group.world_size());
    // The first candidate is the historical stream-per-peer schedule, so a host
    // that prefers it -- an NVLink box, where each peer has its own link --
    // keeps it.
    std::vector<CESchedule> candidates{{std::max(ws - 1, 1), false}};
    if (ws > 2)
        candidates.push_back({1, false});
    if (ws > 3)
        candidates.push_back({2, false});
    if (layered_possible) {
        candidates.push_back({1, true});
        candidates.push_back({2, true});
    }
    std::vector<float> times;
    times.reserve(candidates.size());
    for (const auto& candidate : candidates)
        times.push_back(microbench_us(
            [&] {
                launch(candidate);
                barrier();
            },
            stream));
    if (ws > 1)
        reduce_candidate_times(group, times, barrier, stream);
    const size_t     winner = std::min_element(times.begin(), times.end()) - times.begin();
    const CESchedule best   = candidates[winner];
    if (group.rank() == 0 && std::getenv("FAST_ULYSSES_CE_TUNE_VERBOSE") != nullptr) {
        std::cout << "[ulysses CE tune] ws=" << ws << " -> remote_streams=" << best.remote_streams
                  << " layered=" << (best.layered ? 1 : 0) << " | " << times[winner] << " us/call |";
        // The candidates the winner beat, so a regression on a new host can be
        // read off one run instead of rebuilt from guesses.
        for (size_t index = 0; index < candidates.size(); ++index)
            std::cout << " " << candidates[index].remote_streams << (candidates[index].layered ? "L" : "")
                      << "=" << times[index];
        std::cout << std::endl;
    }
    return best;
}

}  // namespace ulysses
