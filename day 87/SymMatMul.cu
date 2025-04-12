#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void matrixMulKernel(const float* A, const float* B, float* C, size_t n) {
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        float sum = 0.0f;
        for (size_t k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t n) {    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    matrixMulKernel<<<grid, block>>>(input_a, input_b, output_c, n);
    

    cudaDeviceSynchronize();
}
