#include <array>
#include <cstddef>
#include <bit>

#include "cuda_tridiag_kernels.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template <typename DTYPE>
__global__ void TridiagKernel(
    const DTYPE *a,
    const DTYPE *b,
    DTYPE *c,
    DTYPE *d,
    DTYPE *solution,
    int n,
    int num_chunks
){
  for (std::int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_chunks;
       idx += blockDim.x * gridDim.x) {
    DTYPE b0 = b[idx];
    c[idx] /= b0;
    d[idx] /= b0;

    DTYPE norm_factor;
    unsigned int indj = idx;
    DTYPE ai;
    DTYPE cm1;
    DTYPE dm1;

    for (int j = 0; j < n-1; ++j) {
      // c and d from last iteration
      cm1 = c[indj];
      dm1 = d[indj];
      // jump to next chunk
      indj += num_chunks;
      ai = a[indj];
      norm_factor = 1.0f / (b[indj] - ai * cm1);
      c[indj] = c[indj] * norm_factor;
      d[indj] = (d[indj] - ai * dm1) * norm_factor;
    }
    int lastIndx = idx + num_chunks*(n-1);
    solution[lastIndx] = d[lastIndx];
    for (int j=0; j < n-1; ++j) {
      lastIndx -= num_chunks;
      solution[lastIndx] = d[lastIndx] - c[lastIndx] * solution[lastIndx + num_chunks];
    }
  }
}

struct TridiagDescriptor {
  std::int64_t total_size;
  std::int64_t num_systems;
  std::int64_t system_depth;
};

// Unpacks a descriptor object from a byte string.
template <typename T>
const T* UnpackDescriptor(const char* opaque, std::size_t opaque_len) {
  if (opaque_len != sizeof(T)) {
    throw std::runtime_error("Descriptor was not encoded correctly.");
  }
  return reinterpret_cast<const T*>(opaque);
}

template <typename DTYPE>
void CudaTridiag(cudaStream_t stream, void** buffers, const char* opaque,
                      std::size_t opaque_len) {

  const auto& descriptor = *UnpackDescriptor<TridiagDescriptor>(opaque, opaque_len);
  const int num_systems = descriptor.num_systems;
  const int system_depth = descriptor.system_depth;

  const DTYPE* a = reinterpret_cast<const DTYPE*>(buffers[0]);
  const DTYPE* b = reinterpret_cast<const DTYPE*>(buffers[1]);
  DTYPE* c = reinterpret_cast<DTYPE*>(buffers[2]); // should be const
  DTYPE* d = reinterpret_cast<DTYPE*>(buffers[3]); // should be const
  DTYPE* out = reinterpret_cast<DTYPE*>(buffers[4]); // output
  const int BLOCK_SIZE = 128;
  const std::int64_t grid_dim = std::min<std::int64_t>(1024, (num_systems + BLOCK_SIZE - 1) / BLOCK_SIZE);
  TridiagKernel<DTYPE><<<grid_dim, BLOCK_SIZE, 0, stream>>>(a, b, c, d, out, system_depth, num_systems);
  gpuErrchk(cudaPeekAtLastError());
}

void CudaTridiagFloat(cudaStream_t stream, void** buffers, const char* opaque,
                      std::size_t opaque_len) {
  CudaTridiag<float>(stream, buffers, opaque, opaque_len);
}
void CudaTridiagDouble(cudaStream_t stream, void** buffers, const char* opaque,
                      std::size_t opaque_len) {
  CudaTridiag<double>(stream, buffers, opaque, opaque_len);
}
