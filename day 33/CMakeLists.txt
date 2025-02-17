cmake_minimum_required(VERSION 3.10)
project(matrix_mult_comparison LANGUAGES CXX HIP)

# Find HIP package
find_package(HIP REQUIRED)
find_package(rocblas REQUIRED)
find_package(rocsparse REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set HIP flags
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -fPIC")
set(CMAKE_HIP_STANDARD 17)

# Add executable
add_executable(matrix_mult_comparison
    main.cpp
    sparse_gemm.cpp
    strassen.cpp
    winograd.cpp
    utils.cpp
)

# Set source file properties
set_source_files_properties(sparse_gemm.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(strassen.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(winograd.cpp PROPERTIES LANGUAGE HIP)

# Include directories
target_include_directories(matrix_mult_comparison PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${rocblas_INCLUDE_DIRS}
    ${rocsparse_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(matrix_mult_comparison PRIVATE
    hip::host
    roc::rocblas
    roc::rocsparse
)

# Set compile options
target_compile_options(matrix_mult_comparison PRIVATE
    -Wall
    -Wextra
    -O3
) 