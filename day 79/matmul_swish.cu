#include <cuda_runtime.h>
#include <math.h>

__global__ void compute_kernel(const float* input_matrix, const float* weight_matrix, const float* bias, float scaling_factor, float* output, size_t batch_size, size_t in_features, size_t out_features) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < batch_size && col < out_features) {
        float sum = 0.0f;
        for (int k = 0; k < in_features; ++k) {
            sum += input_matrix[row * in_features + k] * weight_matrix[col * in_features + k];
        }
        sum += bias[col];
        float sigmoid = 1.0f / (1.0f + expf(-sum));
        float swish = sum * sigmoid;
        output[row * out_features + col] = scaling_factor * swish;
    }
}

extern "C" void solution(const float* input_matrix, const float* weight_matrix, const float* bias, float scaling_factor, float* output, size_t batch_size, size_t in_features, size_t out_features) {
    dim3 block(16, 16);
    dim3 grid(
        (batch_size + block.x - 1) / block.x,
        (out_features + block.y - 1) / block.y
    );
    compute_kernel<<<grid, block>>>(input_matrix, weight_matrix, bias, scaling_factor, output, batch_size, in_features, out_features);
}