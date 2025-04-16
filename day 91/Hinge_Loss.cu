#include <cuda_runtime.h>


__global__ void hingeKernel(const float* predictions, const float* targets, float* output, size_t n) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float prod = predictions[idx] * targets[idx];
        output[idx] = fmaxf(0.0f, 1.0f - prod);
    }
}


extern "C" void solution(const float* predictions, const float* targets, float* output, size_t n) {
    // I found this to be the best configuration for the kernel (h100)
    const int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;
    
    hingeKernel<<<gridSize, blockSize>>>(predictions, targets, output, n);
    
  
}