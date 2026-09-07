#pragma once

#include "ulysses_group.cuh"
#include <ATen/ATen.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <functional>
#include <string>
#include <torch/custom_class.h>
#include <tuple>
#include <vector>

namespace ulysses {

at::Tensor logical_subgroup_all_to_all_single_4d_ce(
    const c10::intrusive_ptr<UlyssesGroup>& group,
    at::Tensor                              input,
    std::vector<int64_t>                    peer_ranks,
    int64_t                                 mode,
    std::string                             tag,
    int64_t                                 storage_slots,
    int64_t                                 storage_index);

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
                      cudaStream_t                 stream);

// Copy-engine all-gather. `schedule` decides how many remote destinations may
// be in flight and whether the socket boundary is crossed once and relayed;
// `layout` names the peers a relay may go through; `phase_barrier` is only
// invoked by the layered schedule, between the exchange and the relay, and must
// be collective across the group.
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
                         cudaStream_t                 stream);

std::tuple<at::Tensor, at::Tensor> all_gather_kv_4d(
    const c10::intrusive_ptr<UlyssesGroup>& group,
    at::Tensor                              key,
    at::Tensor                              value,
    std::string                             key_tag,
    std::string                             value_tag,
    bool                                    use_ce);

// The trailing handles are the peer and local addresses every later phase
// reuses: peer key, peer value, peer arrivals, local acks, and peer acks.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, std::vector<int64_t>>
ring_ce_prepare_kv_4d(
    const c10::intrusive_ptr<UlyssesGroup>& group,
    at::Tensor                              key,
    at::Tensor                              value,
    std::string                             key_tag,
    std::string                             value_tag,
    std::string                             state_tag,
    int64_t                                 epoch);

void ring_ce_forward_kv_4d(
    const c10::intrusive_ptr<UlyssesGroup>& group,
    at::Tensor                              landing_key,
    at::Tensor                              landing_value,
    int64_t                                 peer_key_ptr,
    int64_t                                 peer_value_ptr,
    int64_t                                 peer_arrival_ptr,
    int64_t                                 local_ack_ptr,
    int64_t                                 epoch,
    int64_t                                 phase);

at::Tensor ring_ce_wait_ready(
    at::Tensor arrivals,
    int64_t   world_size,
    int64_t   epoch,
    int64_t   phase);

void ring_ce_publish_consumed(
    const c10::intrusive_ptr<UlyssesGroup>& group,
    int64_t                                 peer_ack_ptr,
    int64_t                                 device_index,
    int64_t                                 epoch,
    int64_t                                 phase);

at::Tensor logical_subgroup_barrier(
    const c10::intrusive_ptr<UlyssesGroup>& group,
    std::vector<int64_t>                    peer_ranks,
    std::string                             tag,
    int64_t                                 epoch);

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
    int64_t                                 remote_blocks);

}  // namespace ulysses
