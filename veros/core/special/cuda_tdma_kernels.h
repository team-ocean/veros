#pragma once

#include <cuda_runtime.h>

cudaError_t CudaTridiagFloat(cudaStream_t stream, int num_systems, int system_depth,
                            const float* a, const float* b, const float* c, const float* d,
                            float* out, float* workspace);
cudaError_t CudaTridiagDouble(cudaStream_t stream, int num_systems, int system_depth,
                             const double* a, const double* b, const double* c, const double* d,
                             double* out, double* workspace);
