cmake_minimum_required(VERSION 3.18)
project(sparse_ffn LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Target Ampere and newer (sm_80+), also include sm_86, sm_89, sm_90
set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)

# Release build flags
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")

include_directories(src)

# Static library with all kernels
add_library(sparse_ffn STATIC
  src/dense_matvec.cu
  src/silu_threshold.cu
  src/sparse_matvec.cu
  src/dense_ffn.cu
  src/sparse_ffn.cu
)

set_target_properties(sparse_ffn PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

# Correctness test executable
add_executable(test_correctness tests/test_correctness.cu)
target_link_libraries(test_correctness sparse_ffn)
set_target_properties(test_correctness PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Latency benchmark executable
add_executable(bench_latency benchmarks/bench_latency.cu)
target_link_libraries(bench_latency sparse_ffn)
set_target_properties(bench_latency PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
