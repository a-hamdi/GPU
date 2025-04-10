#include <cuda_runtime.h>


__global__ void tensorMatrixMultKernel(
    const float* A,
    const float* B,
    float* C,
    size_t B_dim,
    size_t I_dim,
    size_t J_dim,
    size_t L_dim,
    size_t K_dim
) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    

    int total_elements = B_dim * I_dim * J_dim * K_dim;
    if (idx < total_elements) {
        int k = idx % K_dim;
        int j = (idx / K_dim) % J_dim;
        int i = (idx / (K_dim * J_dim)) % I_dim;
        int b = idx / (K_dim * J_dim * I_dim);
        
    
        size_t c_idx = ((b * I_dim + i) * J_dim + j) * K_dim + k;
        
   
        float sum = 0.0f;
        

        size_t a_base = ((b * I_dim + i) * J_dim + j) * L_dim;
       
        for (int l = 0; l < L_dim; l++) {
            sum += A[a_base + l] * B[l * K_dim + k];
        }

        C[c_idx] = sum;
    }
}

extern "C" void solution(const float* A, const float* B, float* C, size_t b, size_t i, size_t j, size_t l, size_t k) {

    size_t total_elements = b * i * j * k;
    

    int threadsPerBlock = 256;
    int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
    

    tensorMatrixMultKernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, b, i, j, l, k);
}