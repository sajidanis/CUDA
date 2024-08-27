cmake_minimum_required(VERSION 3.28)
project(CudaGraph CUDA)

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CUDA_STANDARD 17)

include_directories(${CMAKE_SOURCE_DIR}/reader)
include_directories(${CMAKE_SOURCE_DIR}/algorithms)

add_executable(ParallelGraph 
        main.cu
        reader/marketReader.cuh
        algorithms/mis.cuh
)

set_target_properties(ParallelGraph PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_property(TARGET ParallelGraph PROPERTY CUDA_ARCHITECTURES 61-real)