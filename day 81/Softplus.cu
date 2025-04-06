#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ float softplus(float x) {
    return log1p(expf(x)); // log1p(x) = log(1 + x), numerically stable
}

__global__ void softplus_kernel(const float* input, float* output, size_t n, size_t m) {
    // Flattened index for 2D matrix
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = n * m;

    if (idx < total_elements) {
        output[idx] = softplus(input[idx]);
    }
}

extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
    size_t total_elements = n * m;
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;

    softplus_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, n, m);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}
