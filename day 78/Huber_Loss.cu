#include <cuda_runtime.h>
#include <cmath>

__global__ void huber_loss_kernel(const float *predictions, const float *targets, float *output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < n; i += stride) {
        float diff = predictions[i] - targets[i];
        float abs_diff = fabsf(diff);
        if (abs_diff < 1.0f)
            output[i] = 0.5f * diff * diff;
        else
            output[i] = abs_diff - 0.5f;
    }
}

extern "C" void solution(const float* predictions, const float* targets, float* output, size_t n) {
   
    int threadsPerBlock = 256;
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    

    huber_loss_kernel<<<numBlocks, threadsPerBlock>>>(predictions, targets, output, n);
    

    cudaDeviceSynchronize();
}
