#pragma once

#include <cstdint>
#include <limits>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

template <typename T>
using CpuKernel = void (*)(int64_t, int64_t, T*, T*, T*, T*, int32_t*, T*, T*);
static CpuKernel<float> float_kernel;
static CpuKernel<double> double_kernel;

inline void SetTDMACpuKernels(void* float_target, void* double_target) {
  float_kernel = reinterpret_cast<CpuKernel<float>>(float_target);
  double_kernel = reinterpret_cast<CpuKernel<double>>(double_target);
}

template <ffi::DataType T>
ffi::Error TDMACpu(ffi::Buffer<T> a, ffi::Buffer<T> b, ffi::Buffer<T> c,
                   ffi::Buffer<T> d, ffi::Buffer<ffi::S32> depths,
                   ffi::ResultBuffer<T> out, ffi::ResultBuffer<T> workspace) {
  auto dims = a.dimensions();
  if (dims.size() == 0 || dims.back() <= 0 || dims.back() > std::numeric_limits<int32_t>::max())
    return ffi::Error::InvalidArgument("TDMA requires a positive int32 depth");
  int64_t stride = dims.back();
  int64_t systems = a.element_count() / stride;
  if (!(b.dimensions() == dims) || !(c.dimensions() == dims) || !(d.dimensions() == dims) ||
      !(out->dimensions() == dims) || depths.element_count() != systems ||
      workspace->element_count() < stride)
    return ffi::Error::InvalidArgument("TDMA buffer shapes do not match");
  for (int64_t i = 0; i < systems; ++i)
    if (depths.typed_data()[i] < 0 || depths.typed_data()[i] > stride)
      return ffi::Error::InvalidArgument("TDMA wet depth exceeds column depth");
  if constexpr (T == ffi::F64)
    double_kernel(systems, stride, a.typed_data(), b.typed_data(), c.typed_data(),
                  d.typed_data(), depths.typed_data(), out->typed_data(), workspace->typed_data());
  else
    float_kernel(systems, stride, a.typed_data(), b.typed_data(), c.typed_data(),
                 d.typed_data(), depths.typed_data(), out->typed_data(), workspace->typed_data());
  return ffi::Error::Success();
}

#define CPU_HANDLER(Name, Type) \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(Name, TDMACpu<Type>, \
    ffi::Ffi::Bind().Arg<ffi::Buffer<Type>>().Arg<ffi::Buffer<Type>>() \
      .Arg<ffi::Buffer<Type>>().Arg<ffi::Buffer<Type>>().Arg<ffi::Buffer<ffi::S32>>() \
      .Ret<ffi::Buffer<Type>>().Ret<ffi::Buffer<Type>>())
CPU_HANDLER(TDMACpuFloat, ffi::F32);
CPU_HANDLER(TDMACpuDouble, ffi::F64);
#undef CPU_HANDLER
