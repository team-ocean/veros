#include <nanobind/nanobind.h>
#include "tdma_cuda_ffi.h"

namespace nb = nanobind;

NB_MODULE(tdma_cuda_, module) {
  nb::dict targets;
  targets["tdma_cuda_float"] = nb::capsule(reinterpret_cast<void*>(TDMACudaFloat), "xla._CUSTOM_CALL_TARGET");
  targets["tdma_cuda_double"] = nb::capsule(reinterpret_cast<void*>(TDMACudaDouble), "xla._CUSTOM_CALL_TARGET");
  module.attr("gpu_ffi_targets") = targets;
}
