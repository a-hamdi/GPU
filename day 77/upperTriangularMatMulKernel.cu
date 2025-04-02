#include <cuda_runtime.h>

__global__ void upperTriangularMatMulKernel(const float* A, const float* B, float* C, size_t n) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        if (row <= col) {
            float sum = 0.0f;
            
            for (int k = row; k <= col; ++k) {
                sum += A[row * n + k] * B[k * n + col];
            }

            C[row * n + col] = sum;
        } else {
          
            C[row * n + col] = 0.0f;
            
        }
    }
}

extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t n) {
    
    dim3 blockDim(16, 16);
  
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);

  
    upperTriangularMatMulKernel<<<gridDim, blockDim>>>(input_a, input_b, output_c, n);

  
    cudaDeviceSynchronize();
}
