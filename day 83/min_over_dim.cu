#include <cuda_runtime.h>
#include <cfloat>  

__global__ void minReduceKernel(const float* input, float* output, 
                              size_t* shape, size_t ndim, int dim,
                              size_t before_size, size_t dim_size, size_t after_size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t output_size = before_size * after_size;
    
    if (idx >= output_size) return;
    
    size_t before_idx = idx / after_size;
    size_t after_idx = idx % after_size;
    
    float min_val = FLT_MAX;
    for (size_t d = 0; d < dim_size; d++) {
        size_t input_idx = (before_idx * dim_size + d) * after_size + after_idx;
        min_val = min(min_val, input[input_idx]);
    }
    
    output[idx] = min_val;
}

extern "C" void solution(const float* input, int dim, float* output, size_t* shape, size_t ndim) {
    size_t before_size = 1;
    for (int i = 0; i < dim; i++) {
        before_size *= shape[i];
    }
    
    size_t dim_size = shape[dim];
    
    size_t after_size = 1;
    for (int i = dim + 1; i < ndim; i++) {
        after_size *= shape[i];
    }
    
    size_t output_size = before_size * after_size;
    

    int blockSize = 256;
    int numBlocks = (output_size + blockSize - 1) / blockSize;
    
    minReduceKernel<<<numBlocks, blockSize>>>(
        input, output, shape, ndim, dim,
        before_size, dim_size, after_size
    );
}