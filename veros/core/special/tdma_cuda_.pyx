# cython: language=c++

from cpython.pycapsule cimport PyCapsule_New


cdef extern from "tdma_cuda_ffi.h":
    ctypedef struct XLA_FFI_CallFrame:
        pass
    ctypedef struct XLA_FFI_Error:
        pass
    XLA_FFI_Error* TDMACudaFloat(XLA_FFI_CallFrame*)
    XLA_FFI_Error* TDMACudaDouble(XLA_FFI_CallFrame*)


gpu_ffi_targets = {
    "tdma_cuda_float": PyCapsule_New(<void*>TDMACudaFloat, "xla._CUSTOM_CALL_TARGET", NULL),
    "tdma_cuda_double": PyCapsule_New(<void*>TDMACudaDouble, "xla._CUSTOM_CALL_TARGET", NULL),
}
