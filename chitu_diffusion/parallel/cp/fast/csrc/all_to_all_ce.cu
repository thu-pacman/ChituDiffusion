// CE (copy-engine) transfer path for the uniform 4D all-to-all: pitched
// cudaMemcpy2DAsync fan-out over the CE streams the tuned schedule asks for, joined back
// to the launching stream with events. The data movement uses DMA engines and zero SMs,
// so -- unlike the SM-resident scatter or the (1-block) TMA kernel, which cannot get a
// block slot while e.g. cuBLAS nvjet GEMMs hold every SM -- it runs concurrently with
// compute. Measured (exclusive 4xH100/4xH200, Wan ws=4): standalone ~0.69ms
// (209 GB/s, vs 385 GB/s for a bare pitched peer memcpy), but 93-94% of it overlaps a
// concurrent GEMM chain where the kernel paths overlap ~25-38% -- net exposed time
// ~0.05ms/call vs ~0.36-0.38ms.
//
// How many destinations may be in flight is a property of the host's interconnect, not a
// constant: an NVLink GPU has an independent link per peer and wants them all open, while
// a PCIe GPU shares one egress port and loses efficiency for every extra destination it
// splits it across. The schedule is therefore measured per shape by the shared CE tuner
// (ce_schedule.cuh) rather than fixed here.
//
// Per-peer transfer shape (mode0): destination peer p receives, for every (b, s) row,
// the contiguous n_local*d block at head-column p: s_local rows of n_local*d*elem bytes,
// source pitch n_global*d*elem, contiguous destination rows -> one pitched 2D copy per
// (peer, b). mode1 is the inverse (contiguous source rows, pitched destination).
// Addressing mirrors a2a_copy_generic (all_to_all.cu) byte for byte.
//
// NOTE (measured on CUDA 13.3, exclusive GPUs; alternatives tried and REVERTED):
// - one single stream for everything: the local copy loses its overlap with the remote
//   copies -- 0.82ms standalone, no upside elsewhere. (Serializing only the REMOTE
//   copies while the local one keeps its own stream is a schedule candidate, and the
//   one a PCIe host picks.)
// - cudaMemcpy3DBatchAsync: the driver's pitched batch path is itself slow (0.82ms,
//   and 1.35ms with cudaMemcpyFlagPreferOverlapWithCompute); it also rejects the
//   LEGACY default stream with "invalid argument" (explicit streams required).
#include "ulysses_group.cuh"

#include <algorithm>

namespace ulysses {

namespace {

// One pitched 2D copy: (peer p, batch i) -> pointers and pitches in bytes.
struct CopyOp {
    const uint8_t* src;
    uint8_t*       dst;
    int64_t        spitch, dpitch;
};

CopyOp make_op(const void* src, uint64_t peer_ptr, const Ulysses4DDims& dims, int mode, int elem_size, int p, int i)
{
    const int64_t row_w  = static_cast<int64_t>(dims.n_local) * dims.d * elem_size;
    const int64_t pitchg = static_cast<int64_t>(dims.n_global) * dims.d * elem_size;
    CopyOp        op;
    if (mode == 0) {
        op.src = static_cast<const uint8_t*>(src) + static_cast<int64_t>(i) * dims.s_local * pitchg
                 + static_cast<int64_t>(p) * row_w;
        op.spitch = pitchg;
        op.dst    = reinterpret_cast<uint8_t*>(peer_ptr)
                 + (static_cast<int64_t>(i) * dims.s_global + static_cast<int64_t>(dims.rank) * dims.s_local) * row_w;
        op.dpitch = row_w;
    }
    else {
        op.src = static_cast<const uint8_t*>(src)
                 + (static_cast<int64_t>(i) * dims.s_global + static_cast<int64_t>(p) * dims.s_local) * row_w;
        op.spitch = row_w;
        op.dst    = reinterpret_cast<uint8_t*>(peer_ptr) + static_cast<int64_t>(i) * dims.s_local * pitchg
                 + static_cast<int64_t>(dims.rank) * row_w;
        op.dpitch = pitchg;
    }
    return op;
}

}  // namespace

void launch_a2a_ce(const void*                  src,
                   const std::vector<uint64_t>& peer_ptrs,
                   const Ulysses4DDims&         dims,
                   int                          mode,
                   int                          elem_size,
                   const CESchedule&            schedule,
                   const CEResources&           ce,
                   cudaStream_t                 stream)
{
    const int     ws    = static_cast<int>(peer_ptrs.size());
    const int64_t row_w = static_cast<int64_t>(dims.n_local) * dims.d * elem_size;
    // ce.streams holds one stream per rank; the schedule uses a prefix of them for the
    // remote destinations and the next one for the local copy, which moves through HBM
    // instead of the egress port and so should not queue behind a remote transfer.
    const int remote_streams = std::max(1, std::min(schedule.remote_streams, std::max(ws - 1, 1)));
    const int local_slot     = ws > 1 ? remote_streams : 0;
    const int used_slots     = local_slot + 1;

    fork_ce_streams(ce, used_slots, stream);
    int remote = 0;
    for (int step = 0; step < ws; ++step) {
        // Staggered visit order (ce_schedule.cuh): submitting peers in rank order makes
        // every rank open its first transfer against the same receiver, and a receiver
        // serves one sender at a time.
        const int          p    = ce_peer_at(dims.rank, step, ws);
        const int          slot = p == dims.rank ? local_slot : remote++ % remote_streams;
        const cudaStream_t cs   = ce.streams[slot];
        for (int i = 0; i < dims.b; ++i) {
            const CopyOp op = make_op(src, peer_ptrs[p], dims, mode, elem_size, p, i);
            ULYSSES_CUDA_CHECK(
                cudaMemcpy2DAsync(op.dst, op.dpitch, op.src, op.spitch, row_w, dims.s_local, cudaMemcpyDefault, cs));
        }
    }
    join_ce_streams(ce, used_slots, stream);
}

}  // namespace ulysses
