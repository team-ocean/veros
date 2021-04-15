#pragma once
#include <cstddef>
#include <string>

#include <cuda_runtime.h>

std::string BuildCudaTridiagDescriptor(std::int64_t total_size, std::int64_t num_systems, std::int64_t system_depth);

void CudaTridiagFloat(cudaStream_t stream, void** buffers, const char* opaque,
                      std::size_t opaque_len);
void CudaTridiagDouble(cudaStream_t stream, void** buffers, const char* opaque,
                      std::size_t opaque_len);

