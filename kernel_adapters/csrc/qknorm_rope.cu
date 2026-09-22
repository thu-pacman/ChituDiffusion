// SPDX-License-Identifier: Apache-2.0
//
// Adapted from SGLang's diffusion QKNorm+RoPE JIT kernel:
// https://github.com/sgl-project/sglang/blob/d6ef68881e26/python/sglang/jit_kernel/csrc/diffusion/qknorm_rope.cuh
// Copyright 2023-2024 SGLang Team.
//
// Changes: remove sgl-kernel/TVM dependencies and PDL, use the PyTorch
// extension API, and accept Flux2/diffusers' exact per-token multi-axis
// cos/sin tensors directly. Thanks to the SGLang community.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

constexpr int kHeadDim = 128;
constexpr int kThreads = 256;

template <typename scalar_t>
__device__ __forceinline__ float scalar_to_float(scalar_t value);

template <>
__device__ __forceinline__ float scalar_to_float<__half>(__half value) {
  return __half2float(value);
}

template <>
__device__ __forceinline__ float scalar_to_float<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t float_to_scalar(float value);

template <>
__device__ __forceinline__ __half float_to_scalar<__half>(float value) {
  return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 float_to_scalar<__nv_bfloat16>(float value) {
  return __float2bfloat16_rn(value);
}

template <typename scalar_t>
__global__ void fused_qknorm_rope_kernel(
    scalar_t* __restrict__ q,
    scalar_t* __restrict__ k,
    const scalar_t* __restrict__ q_weight,
    const scalar_t* __restrict__ k_weight,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    int64_t batch,
    int64_t sequence,
    int64_t heads,
    float eps) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int64_t total_heads = batch * sequence * heads;
  int64_t work = static_cast<int64_t>(blockIdx.x) * (kThreads / 32) + warp;

  for (; work < total_heads; work += static_cast<int64_t>(gridDim.x) * (kThreads / 32)) {
    const int64_t token = (work / heads) % sequence;
    const int64_t base = work * kHeadDim;
    float q_values[4];
    float k_values[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dim = lane * 4 + i;
      q_values[i] = scalar_to_float(q[base + dim]);
      k_values[i] = scalar_to_float(k[base + dim]);
    }

    float q_sum = 0.0f;
    float k_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      q_sum += q_values[i] * q_values[i];
      k_sum += k_values[i] * k_values[i];
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      q_sum += __shfl_down_sync(0xffffffff, q_sum, offset);
      k_sum += __shfl_down_sync(0xffffffff, k_sum, offset);
    }
    q_sum = __shfl_sync(0xffffffff, q_sum, 0);
    k_sum = __shfl_sync(0xffffffff, k_sum, 0);
    const float q_scale = rsqrtf(q_sum / kHeadDim + eps);
    const float k_scale = rsqrtf(k_sum / kHeadDim + eps);
    const int64_t rope_base = token * kHeadDim;

#pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
      const int first = lane * 4 + pair * 2;
      const float c = cos[rope_base + first];
      const float s = sin[rope_base + first];
      // PyTorch RMSNorm materializes an FP16/BF16 tensor before diffusers
      // applies RoPE in FP32. Preserve that intermediate rounding policy.
      const float qx = scalar_to_float(float_to_scalar<scalar_t>(
          q_values[pair * 2] * q_scale * scalar_to_float(q_weight[first])));
      const float qy = scalar_to_float(float_to_scalar<scalar_t>(
          q_values[pair * 2 + 1] * q_scale * scalar_to_float(q_weight[first + 1])));
      const float kx = scalar_to_float(float_to_scalar<scalar_t>(
          k_values[pair * 2] * k_scale * scalar_to_float(k_weight[first])));
      const float ky = scalar_to_float(float_to_scalar<scalar_t>(
          k_values[pair * 2 + 1] * k_scale * scalar_to_float(k_weight[first + 1])));
      q[base + first] = float_to_scalar<scalar_t>(qx * c - qy * s);
      q[base + first + 1] = float_to_scalar<scalar_t>(qy * c + qx * s);
      k[base + first] = float_to_scalar<scalar_t>(kx * c - ky * s);
      k[base + first + 1] = float_to_scalar<scalar_t>(ky * c + kx * s);
    }
  }
}

void fused_qknorm_rope(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    torch::Tensor cos,
    torch::Tensor sin,
    double eps) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda(), "q and k must be CUDA tensors");
  TORCH_CHECK(q.device() == k.device() && q.device() == q_weight.device() &&
                  q.device() == k_weight.device() && q.device() == cos.device() &&
                  q.device() == sin.device(),
              "all inputs must be on q's CUDA device");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous(), "q and k must be contiguous");
  TORCH_CHECK(q_weight.is_contiguous() && k_weight.is_contiguous() && cos.is_contiguous() &&
                  sin.is_contiguous(),
              "weights, cos, and sin must be contiguous");
  TORCH_CHECK(q.sizes() == k.sizes() && q.dim() == 4, "q and k must have equal BSHD shapes");
  TORCH_CHECK(q.size(3) == kHeadDim, "only head_dim=128 is supported");
  TORCH_CHECK(q.scalar_type() == k.scalar_type(), "q and k dtypes must match");
  TORCH_CHECK(
      q.scalar_type() == torch::kFloat16 || q.scalar_type() == torch::kBFloat16,
      "only FP16/BF16 is supported");
  TORCH_CHECK(q_weight.dim() == 1 && k_weight.dim() == 1 &&
                  q_weight.size(0) == kHeadDim && k_weight.size(0) == kHeadDim &&
                  q_weight.scalar_type() == q.scalar_type() &&
                  k_weight.scalar_type() == q.scalar_type(),
              "weights must be [128] and match q dtype");
  TORCH_CHECK(cos.scalar_type() == torch::kFloat32 && sin.scalar_type() == torch::kFloat32,
              "cos and sin must be float32");
  TORCH_CHECK(cos.sizes() == sin.sizes() && cos.dim() == 2 && cos.size(0) == q.size(1) &&
                  cos.size(1) == kHeadDim,
              "cos/sin must be [sequence, 128]");

  const auto blocks = std::min<int64_t>((q.size(0) * q.size(1) * q.size(2) + 7) / 8, 4096);
  const auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
  if (q.scalar_type() == torch::kFloat16) {
    fused_qknorm_rope_kernel<__half><<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<__half*>(q.data_ptr()),
        reinterpret_cast<__half*>(k.data_ptr()),
        reinterpret_cast<const __half*>(q_weight.data_ptr()),
        reinterpret_cast<const __half*>(k_weight.data_ptr()),
        cos.data_ptr<float>(), sin.data_ptr<float>(), q.size(0), q.size(1), q.size(2), static_cast<float>(eps));
  } else {
    fused_qknorm_rope_kernel<__nv_bfloat16><<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(k.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(q_weight.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k_weight.data_ptr()),
        cos.data_ptr<float>(), sin.data_ptr<float>(), q.size(0), q.size(1), q.size(2), static_cast<float>(eps));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_qknorm_rope", &fused_qknorm_rope, "Flux2 fused QK RMSNorm + RoPE");
}
