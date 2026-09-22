#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/library.h>
#include <vector>

namespace {

// Resolve a kernel from the actual fatbinary without launching or initializing
// NVSHMEM. Importing a CUDA extension alone does not establish GPU compatibility.
__global__ void fast_cp_image_probe() {}

std::vector<int64_t> fast_cp_device_info(int64_t device)
{
    const c10::cuda::CUDAGuard guard{static_cast<c10::DeviceIndex>(device)};
    cudaFuncAttributes attributes{};
    const auto status = cudaFuncGetAttributes(&attributes, fast_cp_image_probe);
    if (status != cudaSuccess) {
        cudaGetLastError();
        TORCH_CHECK(false, "Fast CP has no loadable CUDA image on device ", device,
                    ": ", cudaGetErrorString(status),
                    ". Rebuild with tools/install/build_fast_cp.py --architectures 90 120.");
    }
    cudaDeviceProp properties{};
    const auto result = cudaGetDeviceProperties(&properties, static_cast<int>(device));
    TORCH_CHECK(result == cudaSuccess, cudaGetErrorString(result));
    return {properties.major * 10 + properties.minor,
            attributes.binaryVersion,
            attributes.ptxVersion,
            static_cast<int64_t>(properties.sharedMemPerBlockOptin)};
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(fast_ulysses, m)
{
    m.def("fast_cp_device_info(int device) -> int[]");
    m.impl("fast_cp_device_info", c10::DispatchKey::CompositeExplicitAutograd,
           &fast_cp_device_info);
}
