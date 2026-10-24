#pragma once

#include <limits>
#include "cuda_tdma_kernels.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

template <ffi::DataType T>
ffi::Error TDMACuda(cudaStream_t stream, ffi::Buffer<T> a, ffi::Buffer<T> b,
                    ffi::Buffer<T> c, ffi::Buffer<T> d,
                    ffi::ResultBuffer<T> out, ffi::ResultBuffer<T> workspace) {
  auto dims = a.dimensions();
  if (dims.size() == 0 || dims.back() <= 0 || dims.back() > std::numeric_limits<int>::max())
    return ffi::Error::InvalidArgument("TDMA requires a positive int32 depth");
  int64_t systems = a.element_count() / dims.back();
  if (systems > std::numeric_limits<int>::max() ||
      (systems && dims.back() > std::numeric_limits<int>::max() / systems))
    return ffi::Error::InvalidArgument("CUDA TDMA exceeds int32 indexing capacity");
  if (!(b.dimensions() == dims) || !(c.dimensions() == dims) || !(d.dimensions() == dims) ||
      !(out->dimensions() == dims) || !(workspace->dimensions() == dims))
    return ffi::Error::InvalidArgument("TDMA buffer shapes do not match");
  cudaError_t status;
  if constexpr (T == ffi::F64)
    status = CudaTridiagDouble(stream, systems, dims.back(), a.typed_data(), b.typed_data(),
                               c.typed_data(), d.typed_data(), out->typed_data(), workspace->typed_data());
  else
    status = CudaTridiagFloat(stream, systems, dims.back(), a.typed_data(), b.typed_data(),
                              c.typed_data(), d.typed_data(), out->typed_data(), workspace->typed_data());
  if (status != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(status));
  return ffi::Error::Success();
}

#define CUDA_HANDLER(Name, Type) \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(Name, TDMACuda<Type>, \
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>() \
      .Arg<ffi::Buffer<Type>>().Arg<ffi::Buffer<Type>>().Arg<ffi::Buffer<Type>>() \
      .Arg<ffi::Buffer<Type>>().Ret<ffi::Buffer<Type>>().Ret<ffi::Buffer<Type>>())
CUDA_HANDLER(TDMACudaFloat, ffi::F32);
CUDA_HANDLER(TDMACudaDouble, ffi::F64);
#undef CUDA_HANDLER
