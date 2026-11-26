#include <cstdint>
#include <limits>

#include <nanobind/nanobind.h>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
namespace nb = nanobind;

namespace {

template <typename T>
void SolveColumn(int32_t depth, const T* a, const T* b, const T* c,
                 const T* d, T* workspace, T* out) {
  if (depth == 0) return;

  workspace[0] = c[0] / b[0];
  out[0] = d[0] / b[0];
  for (int32_t i = 1; i < depth; ++i) {
    T denom = 1.0 / (b[i] - a[i] * workspace[i - 1]);
    workspace[i] = c[i] * denom;
    out[i] = (d[i] - a[i] * out[i - 1]) * denom;
  }
  for (int32_t i = depth - 2; i >= 0; --i) {
    out[i] -= workspace[i] * out[i + 1];
  }
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

  int64_t offset = 0;
  for (int64_t i = 0; i < systems; ++i) {
    int32_t depth = depths.typed_data()[i];
    int64_t start = offset + stride - depth;
    for (int64_t j = offset; j < start; ++j) {
      out->typed_data()[j] = 0;
    }
    SolveColumn(depth, a.typed_data() + start, b.typed_data() + start,
                c.typed_data() + start, d.typed_data() + start,
                workspace->typed_data(), out->typed_data() + start);
    offset += stride;
  }
  return ffi::Error::Success();
}

}  // namespace

#define CPU_HANDLER(Name, Type) \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(Name, TDMACpu<Type>, \
    ffi::Ffi::Bind().Arg<ffi::Buffer<Type>>().Arg<ffi::Buffer<Type>>() \
      .Arg<ffi::Buffer<Type>>().Arg<ffi::Buffer<Type>>().Arg<ffi::Buffer<ffi::S32>>() \
      .Ret<ffi::Buffer<Type>>().Ret<ffi::Buffer<Type>>())
CPU_HANDLER(TDMACpuFloat, ffi::F32);
CPU_HANDLER(TDMACpuDouble, ffi::F64);
#undef CPU_HANDLER

NB_MODULE(tdma_cpu_, module) {
  nb::dict targets;
  targets["tdma_cpu_float"] = nb::capsule(reinterpret_cast<void*>(TDMACpuFloat), "xla._CUSTOM_CALL_TARGET");
  targets["tdma_cpu_double"] = nb::capsule(reinterpret_cast<void*>(TDMACpuDouble), "xla._CUSTOM_CALL_TARGET");
  module.attr("cpu_ffi_targets") = targets;
}
