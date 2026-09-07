#pragma once
#include <array>
#include <cstdint>
#include <cuda_runtime.h>
#include <functional>
#include <vector>

namespace ulysses {

class UlyssesGroup;

// Peer visit order and stream budget for the copy-engine transfer paths.
//
// Measured on 8x RTX PRO 5000 (no NVLink, two NUMA nodes, PCIe Gen5 x16 per
// GPU), all-gather of a 40.6 MiB K/V shard at ws=8:
//
//   rank-order visit, one stream per peer   38.6 ms   16.8 GiB/s egress
//   staggered visit, one stream per peer    20.2 ms   27.5
//   staggered visit, one remote stream      17.6 ms   31.5
//   staggered visit, layered, one stream    12.4 ms   44.7   (link rate is 51)
//
// Two effects drive that spread:
//
// * Visiting peers in rank order makes every rank push to the same receiver at
//   the same time, and a receiver serves one sender at a time. A staggered
//   order makes each step a permutation of the group, so no receiver is a hot
//   spot. This is worth ~1.9x on its own and costs nothing.
// * A PCIe GPU has ONE egress port shared by every destination, so splitting it
//   across peers only loses efficiency: ~51 GiB/s to a single destination,
//   ~30 spread over three, ~19 spread over seven. An NVLink host has an
//   independent link per peer and wants the opposite, which is why the stream
//   budget is measured per host instead of fixed -- see
//   UlyssesGroup::resolve_ce_schedule.

struct CESchedule {
    // Remote destinations allowed in flight. The local copy always gets a
    // stream of its own: it moves through HBM rather than the egress port the
    // remote transfers queue on, so it should not wait behind them.
    int remote_streams;
    // Cross the NUMA boundary once per rank and replicate inside the node,
    // instead of publishing directly to every peer on the other node. Moves the
    // same bytes out of each GPU but only a quarter of them across the socket
    // interconnect, which saturates at ~98 GiB/s against ~373 GiB/s inside a
    // node. Costs one extra phase barrier, so it only pays for large shards.
    bool layered;
};

// Per-group CE transfer resources: one stream per peer, from which a schedule
// takes the slots it needs. Created lazily by UlyssesGroup::ce_resources(),
// released in destroy(). Serial use only (same contract as the config caches).
// Join events are deliberately NOT pooled here -- see fork_ce_streams.
struct CEResources {
    std::vector<cudaStream_t> streams;
};

// Which transfer a cached schedule belongs to. The all-gather and the two
// all-to-all directions move different per-peer shapes for the same tensor, so
// each tunes on its own key and none of them can collide in the cache.
enum class CEPath : int64_t {
    all_gather        = 0,
    all_to_all_mode_0 = 1,
    all_to_all_mode_1 = 2,
};

// {path, then the five numbers that fix the per-peer transfer shape}.
using CEScheduleKey = std::array<int64_t, 6>;

// Destination for step `step` of a staggered fan-out seen from `rank`. Step 0
// is the rank itself, so the local copy leads and remote steps follow. XOR
// keeps the early steps inside a NUMA half of a power-of-two group; other sizes
// fall back to a rotation, which is also a hot-spot-free permutation.
inline int ce_peer_at(int rank, int step, int world_size)
{
    if ((world_size & (world_size - 1)) == 0)
        return rank ^ step;
    return (rank + step) % world_size;
}

// Which peers the layered schedule may relay through, resolved from the driver
// rather than assumed from the rank order: a host is free to enumerate its GPUs
// across sockets in any order, and pairing the wrong ranks would send every
// relay back over the interconnect the schedule exists to spare.
struct CENumaLayout {
    // NUMA node per rank, -1 where the host does not report one.
    std::vector<int> nodes;
    // Ranks sharing this rank's node, ascending, including this rank.
    std::vector<int> block;
    // This rank's counterpart on the other node, -1 when there is none.
    int partner = -1;
    // Whether the group splits into two equal blocks of at least two ranks,
    // which is what the layered exchange-then-relay pattern needs.
    bool layered = false;
};

// Resolve the layout from a symmetric buffer's peer pointers: each carries the
// device it lives on, and the device's PCI address carries its NUMA node. Every
// rank reads the same host, so every rank derives the same layout.
CENumaLayout resolve_ce_numa_layout(int rank, const std::vector<uint64_t>& peer_ptrs);

// Make the first `slots` CE streams wait for the work already queued on
// `stream`, so the transfers they are about to receive see the inputs.
//
// FRESH events every call -- do not hoist them into CEResources. Re-recording a
// shared event that still has in-flight stream waits (deep enqueue-ahead: many
// deferred barrier=False groups queued behind the device) lets a pending wait
// resolve against a LATER record whose completion depends on this very stream
// progressing -- a circular wait that deadlocks the group (reproduced at ws=2
// with a few undrained groups). Create/destroy is a few us per call and
// depth-safe: the waits capture the dependency at call time, and destroy defers
// until the event retires.
void fork_ce_streams(const CEResources& ce, int slots, cudaStream_t stream);

// Join the first `slots` CE streams back onto `stream`, so the caller's barrier
// publishes only completed transfers.
void join_ce_streams(const CEResources& ce, int slots, cudaStream_t stream);

// Time every candidate fan-out for one transfer shape and keep the fastest.
// `launch` issues the transfer under a candidate schedule and `barrier` is the
// collective handshake that closes one call, so each candidate is timed as the
// real per-call cost. `layered_possible` gates the relay candidates: only the
// all-gather can relay, because there every destination receives the same
// bytes.
//
// Hang safety: the candidate list is a function of world size alone, so under
// SPMD every rank tunes the same sequence and issues the same barriers in the
// same order, and the group agrees on one winner (see reduce_candidate_times).
CESchedule tune_ce_schedule(UlyssesGroup&                                 group,
                            bool                                          layered_possible,
                            const std::function<void(const CESchedule&)>& launch,
                            const std::function<void()>&                  barrier,
                            cudaStream_t                                  stream);

}  // namespace ulysses
