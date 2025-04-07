#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void cosine_similarity_kernel(const float* predictions, const float* targets, float* output, size_t n, size_t d) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float dot = 0.0f;
        float norm_pred = 0.0f;
        float norm_target = 0.0f;
        size_t offset = idx * d;
        
        for (size_t j = 0; j < d; j++) {
            float p = predictions[offset + j];
            float t = targets[offset + j];
            dot += p * t;
            norm_pred += p * p;
            norm_target += t * t;
        }
        
        norm_pred = sqrtf(norm_pred);
        norm_target = sqrtf(norm_target);
        
        const float eps = 1e-8f;
        float denom = fmaxf(eps, norm_pred) * fmaxf(eps, norm_target);
        float cosine_sim = dot / denom;
        
        output[idx] = 1.0f - cosine_sim;
    }
}

extern "C" void solution(const float* predictions, const float* targets, float* output, size_t n, size_t d) {
    size_t total_vectors = n;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_vectors + threadsPerBlock - 1) / threadsPerBlock;
    
    cosine_similarity_kernel<<<blocksPerGrid, threadsPerBlock>>>(predictions, targets, output, n, d);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}
