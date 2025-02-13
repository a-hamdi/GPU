# CUDA Matrix Operations with CUDA Graphs

## Overview
This CUDA program performs a series of matrix operations using GPU acceleration. The program demonstrates the use of CUDA kernels for matrix addition, scaling, squaring, and offsetting while measuring performance with and without CUDA Graphs.

## Features
- **Matrix Addition**: Adds two arrays element-wise.
- **Matrix Scaling**: Multiplies each element of an array by a scalar.
- **Matrix Squaring**: Squares each element of an array.
- **Matrix Offsetting**: Adds a fixed offset to each element of an array.
- **Performance Comparison**: Measures execution time using traditional CUDA execution vs. CUDA Graphs.
- **CUDA Stream and Events**: Uses CUDA streams and events for optimized execution and performance measurement.
- **Result Verification**: Compares GPU-computed results with CPU-verified results to ensure correctness.

## Requirements
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- A C++ compiler with CUDA support (e.g., `nvcc`) 

## Compilation & Execution
### Compilation:
Compile the program using `nvcc`:
```sh
nvcc -O3 -o matrix_operations Cuda_graphs.cu
```
In colab add: -arch=sm_75 for the T4 gpu.
### Running the Program:
Execute the compiled binary:
```sh
./matrix_operations
```

## Explanation of Key Components
### CUDA Kernels
1. **`matrixAdd`**: Computes element-wise addition of two arrays.
2. **`matrixScale`**: Scales each element of an array by a given scalar.
3. **`matrixSquare`**: Squares each element of an array.
4. **`matrixOffset`**: Adds a constant value to each element of an array.

### CUDA Graphs
- The program captures a sequence of GPU operations in a CUDA Graph to reduce kernel launch overhead.
- Traditional execution and CUDA Graph-based execution are timed separately for performance comparison.

### Performance Measurement
- Uses `cudaEventRecord()` to measure execution time.
- Compares traditional execution vs. CUDA Graph execution.

## Expected Output
The program will print execution times for both traditional execution and CUDA Graph execution, followed by a verification message indicating whether GPU results match CPU-computed results.

Example output:
```
Without CUDA Graphs: 220.013 ms
With CUDA Graphs: 89.876 ms
Verification successful! All values match expected result.
``