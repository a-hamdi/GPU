#include <cuda_runtime.h>
#include <cstddef>

#define BLOCK_SIZE 16


__global__ 
void lowerTriangularMultiplyKernel(const float* A, const float* B, float* C, size_t n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        if (col > row) {
            C[row * n + col] = 0.0f;
        } else {
            float sum = 0.0f;
            for (int k = col; k <= row; k++) {
                sum += A[row * n + k] * B[k * n + col];
            }
            C[row * n + col] = sum;
        }
    }
}

extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t n) {
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x,
                 (n + blockDim.y - 1) / blockDim.y);

    lowerTriangularMultiplyKernel<<<gridDim, blockDim>>>(input_a, input_b, output_c, n);

    cudaDeviceSynchronize();
}
