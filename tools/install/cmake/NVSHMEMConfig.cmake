# The PyPI NVSHMEM wheel omits CMake exports. Fast CP uses the host library
# only: its kernels access mapped peer pointers without NVSHMEM device calls.
find_library(CHITU_NVSHMEM_HOST
    NAMES nvshmem_host libnvshmem_host.so.3
    PATHS "${NVSHMEM_HOME}/lib" "${NVSHMEM_HOME}/lib64"
    NO_DEFAULT_PATH REQUIRED)
if(NOT TARGET nvshmem::nvshmem_host)
    add_library(nvshmem::nvshmem_host SHARED IMPORTED)
    set_target_properties(nvshmem::nvshmem_host PROPERTIES
        IMPORTED_LOCATION "${CHITU_NVSHMEM_HOST}"
        INTERFACE_INCLUDE_DIRECTORIES "${NVSHMEM_HOME}/include")
endif()
set(NVSHMEM_FOUND TRUE)
