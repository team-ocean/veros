cmake_minimum_required(VERSION 3.24)
project(veros_tdma LANGUAGES CXX)

option(VEROS_BUILD_ABI3 "Build the CPU extension with the stable Python ABI" OFF)
if(VEROS_BUILD_ABI3)
  find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule)
  set(TDMA_ABI_FLAG STABLE_ABI)
else()
  find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
endif()
find_package(nanobind CONFIG REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(TDMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/veros/core/special")

nanobind_add_module(tdma_cpu_ ${TDMA_ABI_FLAG} NOMINSIZE PROTECT_STACK "${TDMA_DIR}/tdma_cpu.cpp")
target_include_directories(tdma_cpu_ SYSTEM PRIVATE "${JAXLIB_INCLUDE_DIR}")

if(VEROS_BUILD_ABI3)
  return()
endif()

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  find_package(CUDAToolkit QUIET)
endif()

if(CUDAToolkit_FOUND)
  nanobind_add_module(tdma_cuda_ NOMINSIZE PROTECT_STACK
    "${TDMA_DIR}/tdma_cuda.cpp" "${TDMA_DIR}/cuda_tdma_kernels.cu")
  target_include_directories(tdma_cuda_ SYSTEM PRIVATE "${JAXLIB_INCLUDE_DIR}")
  target_link_libraries(tdma_cuda_ PRIVATE CUDA::cudart)
  set_target_properties(tdma_cuda_ PROPERTIES
    CUDA_STANDARD 11
    CUDA_STANDARD_REQUIRED ON
    CUDA_RUNTIME_LIBRARY Shared)
  if(DEFINED ENV{CUDA_COMPUTE_CAPABILITY} AND NOT "$ENV{CUDA_COMPUTE_CAPABILITY}" STREQUAL "")
    set_property(TARGET tdma_cuda_ PROPERTY CUDA_ARCHITECTURES "$ENV{CUDA_COMPUTE_CAPABILITY}")
  endif()
  file(TOUCH "${CMAKE_BINARY_DIR}/cuda-enabled")
endif()
