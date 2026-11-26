#include <algorithm>

#include <cuda_runtime.h>

#include "cuda_tdma_kernels.h"

template <typename DTYPE>
__global__ void TridiagKernel(
    const DTYPE *a,
    const DTYPE *b,
    const DTYPE *c,
    const DTYPE *d,
    DTYPE *cp,
    DTYPE *dp,
    const int num_systems,
    const int system_depth
){
  // TDMA algorithm
  // Solution is written to dp
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_systems;
       idx += blockDim.x * gridDim.x) {

    if (idx >= system_depth * num_systems) {
      return;
    }

    int indj = idx;

    DTYPE denom;
    DTYPE ai;

    DTYPE b0 = b[indj];
    DTYPE cm1 = c[indj] / b0;
    DTYPE dm1 = d[indj] / b0;

    cp[indj] = cm1;
    dp[indj] = dm1;

    // forward pass
    for (int j = 0; j < system_depth-1; ++j) {
      indj += num_systems;
      ai = a[indj];
      denom = 1.0f / (b[indj] - ai * cm1);

      cm1 = c[indj] * denom;
      dm1 = (d[indj] - ai * dm1) * denom;

      cp[indj] = cm1;
      dp[indj] = dm1;
    }

    // backward pass
    for (int j = 0; j < system_depth-1; ++j) {
      indj -= num_systems;
      dp[indj] -= cp[indj] * dp[indj + num_systems];
    }
  }
}

template <typename DTYPE>
cudaError_t CudaTridiag(cudaStream_t stream, int num_systems, int system_depth,
                        const DTYPE* a, const DTYPE* b, const DTYPE* c, const DTYPE* d,
                        DTYPE* out, DTYPE* workspace) {
  if (num_systems == 0) return cudaSuccess;
  const int BLOCK_SIZE = 128;
  const int grid_dim = std::min<int>(1024, (num_systems + BLOCK_SIZE - 1) / BLOCK_SIZE);
  TridiagKernel<DTYPE><<<grid_dim, BLOCK_SIZE, 0, stream>>>(a, b, c, d, workspace, out, num_systems, system_depth);
  return cudaGetLastError();
}

cudaError_t CudaTridiagFloat(cudaStream_t stream, int num_systems, int system_depth,
                            const float* a, const float* b, const float* c, const float* d,
                            float* out, float* workspace) {
  return CudaTridiag<float>(stream, num_systems, system_depth, a, b, c, d, out, workspace);
}

cudaError_t CudaTridiagDouble(cudaStream_t stream, int num_systems, int system_depth,
                             const double* a, const double* b, const double* c, const double* d,
                             double* out, double* workspace) {
  return CudaTridiag<double>(stream, num_systems, system_depth, a, b, c, d, out, workspace);
}
