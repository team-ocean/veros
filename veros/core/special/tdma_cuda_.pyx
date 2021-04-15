# cython: language_level=3

from cpython.pycapsule cimport PyCapsule_New

from libc.stdint cimport int64_t

cdef extern from "cuda_runtime_api.h":
    ctypedef void* cudaStream_t

cdef extern from "cuda_tridiag_kernels.h":
    void CudaTridiagFloat(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len)
    void CudaTridiagDouble(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len)


cdef struct TridiagDescriptor:
    int64_t total_size
    int64_t num_systems
    int64_t system_depth


cpdef bytes build_tridiag_descriptor(int64_t total_size, int64_t num_systems, int64_t system_depth):
    cdef TridiagDescriptor desc = TridiagDescriptor(
        total_size, num_systems, system_depth
    )
    return bytes((<char*> &desc)[:sizeof(TridiagDescriptor)])

gpu_custom_call_targets = {}

cdef register_custom_call_target(fn_name, void* fn):
    cdef const char* name = "xla._CUSTOM_CALL_TARGET"
    gpu_custom_call_targets[fn_name] = PyCapsule_New(fn, name, NULL)

register_custom_call_target(b"tdma_cuda_double", <void*>(CudaTridiagDouble))
register_custom_call_target(b"tdma_cuda_float", <void*>(CudaTridiagFloat))
