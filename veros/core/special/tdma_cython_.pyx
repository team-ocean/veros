# cython: language=c++

import cython
from cpython.pycapsule cimport PyCapsule_New

from libc.stdint cimport int32_t, int64_t


@cython.cdivision(True)
cdef void _tdma_cython_double(int32_t n, double* a, double* b, double* c, double* d, double* cp, double* dp) noexcept nogil:
    cdef:
        int32_t i
        double denom

    if n < 1:
        return

    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]

    for i in range(1, n):
        denom = 1. / (b[i] - a[i] * cp[i - 1])
        cp[i] = c[i] * denom
        dp[i] = (d[i] - a[i] * dp[i - 1]) * denom

    for i in range(n - 2, -1, -1):
        dp[i] -= cp[i] * dp[i + 1]


@cython.cdivision(True)
cdef void _tdma_cython_float(int32_t n, float* a, float* b, float* c, float* d, float* cp, float* dp) noexcept nogil:
    cdef:
        int32_t i
        float denom

    if n < 1:
        return

    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]

    for i in range(1, n):
        denom = 1. / (b[i] - a[i] * cp[i - 1])
        cp[i] = c[i] * denom
        dp[i] = (d[i] - a[i] * dp[i - 1]) * denom

    for i in range(n - 2, -1, -1):
        dp[i] -= cp[i] * dp[i + 1]


cdef void tdma_cython_double(int64_t num_systems, int64_t stride,
        double* a, double* b, double* c, double* d, int32_t* system_depths,
        double* out, double* workspace) noexcept nogil:
    cdef:
        int64_t i, j, system_depth, system_start
        int64_t ii = 0

    for i in range(num_systems):
        system_depth = system_depths[i]
        system_start = stride - system_depth

        for j in range(system_start):
            out[ii + j] = 0.

        _tdma_cython_double(
            system_depth,
            &a[ii + system_start],
            &b[ii + system_start],
            &c[ii + system_start],
            &d[ii + system_start],
            workspace,
            &out[ii + system_start],
        )

        ii += stride


cdef void tdma_cython_float(int64_t num_systems, int64_t stride,
        float* a, float* b, float* c, float* d, int32_t* system_depths,
        float* out, float* workspace) noexcept nogil:
    cdef:
        int64_t i, j, system_depth, system_start
        int64_t ii = 0

    for i in range(num_systems):
        system_depth = system_depths[i]
        system_start = stride - system_depth

        for j in range(system_start):
            out[ii + j] = 0.0

        _tdma_cython_float(
            system_depth,
            &a[ii + system_start],
            &b[ii + system_start],
            &c[ii + system_start],
            &d[ii + system_start],
            workspace,
            &out[ii + system_start],
        )

        ii += stride


cdef extern from "tdma_cpu_ffi.h":
    ctypedef struct XLA_FFI_CallFrame:
        pass
    ctypedef struct XLA_FFI_Error:
        pass
    void SetTDMACpuKernels(void* float_kernel, void* double_kernel)
    XLA_FFI_Error* TDMACpuFloat(XLA_FFI_CallFrame*)
    XLA_FFI_Error* TDMACpuDouble(XLA_FFI_CallFrame*)


SetTDMACpuKernels(<void*>tdma_cython_float, <void*>tdma_cython_double)
cpu_ffi_targets = {
    "tdma_cython_float": PyCapsule_New(<void*>TDMACpuFloat, "xla._CUSTOM_CALL_TARGET", NULL),
    "tdma_cython_double": PyCapsule_New(<void*>TDMACpuDouble, "xla._CUSTOM_CALL_TARGET", NULL),
}
