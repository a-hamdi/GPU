#include <cuda_runtime.h>
#include <cstdio>


__global__ void hard_sigmoid_kernel(const float* input, float* output, size_t total_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return; 

    float x = input[idx];
    if (x <= -3.0f)
        output[idx] = 0.0f;
    else if (x >= 3.0f)
        output[idx] = 1.0f;
    else
        output[idx] = (x + 3.0f) / 6.0f;
}


extern "C" void solution(const float* input, float* output, size_t n, size_t m) {

    size_t total_elements = n * m;
    
    const int threadsPerBlock = 256;
    int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
    
    hard_sigmoid_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, total_elements);
    
  
}
