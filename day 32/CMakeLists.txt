cmake_minimum_required(VERSION 3.21)
project(sgemm_benchmark LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find ROCm packages
find_package(rocblas REQUIRED)

# Set HIP compiler
set(CMAKE_CXX_COMPILER /opt/rocm/bin/hipcc)

# Add executable
add_executable(benchmark_sgemm 
    benchmark_sgemm.cpp
    sgemm.cpp
    kernel3_registers.cpp
)

# Link against rocBLAS
target_link_libraries(benchmark_sgemm PRIVATE 
    roc::rocblas
)

# Set compile options
target_compile_options(benchmark_sgemm PRIVATE 
    -Wall 
    -Wextra
    -O3
    --offload-arch=gfx90a
) 