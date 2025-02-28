#include "solve.h"
#include <cuda_runtime.h>
#include <stdio.h>


__global__ void conv3d_kernel(const float* input, const float* kernel, float* output,
                              int input_depth, int input_rows, int input_cols,
                              int kernel_depth, int kernel_rows, int kernel_cols,
                              int output_depth, int output_rows, int output_cols) {

    int ocol = blockIdx.x * blockDim.x + threadIdx.x;
    int orow = blockIdx.y * blockDim.y + threadIdx.y;
    int od   = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (ocol < output_cols && orow < output_rows && od < output_depth) {
        float sum = 0.0f;

        for (int kd = 0; kd < kernel_depth; kd++) {
            for (int kr = 0; kr < kernel_rows; kr++) {
                for (int kc = 0; kc < kernel_cols; kc++) {

                    int id = od + kd;
                    int ir = orow + kr;
                    int ic = ocol + kc;
                    int input_idx = id * (input_rows * input_cols) + ir * input_cols + ic;
                    int kernel_idx = kd * (kernel_rows * kernel_cols) + kr * kernel_cols + kc;
                    sum += input[input_idx] * kernel[kernel_idx];
                }
            }
        }
        int output_idx = od * (output_rows * output_cols) + orow * output_cols + ocol;
        output[output_idx] = sum;
    }
}

void solve(const float* input, const float* kernel, float* output,
           int input_depth, int input_rows, int input_cols,
           int kernel_depth, int kernel_rows, int kernel_cols) {
    
    int output_depth = input_depth - kernel_depth + 1;
    int output_rows  = input_rows - kernel_rows + 1;
    int output_cols  = input_cols - kernel_cols + 1;
    
    size_t input_size  = input_depth * input_rows * input_cols * sizeof(float);
    size_t kernel_size = kernel_depth * kernel_rows * kernel_cols * sizeof(float);
    size_t output_size = output_depth * output_rows * output_cols * sizeof(float);
    
    float *d_input, *d_kernel, *d_output;
    cudaMalloc((void**)&d_input, input_size);
    cudaMalloc((void**)&d_kernel, kernel_size);
    cudaMalloc((void**)&d_output, output_size);
    
    cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_size, cudaMemcpyHostToDevice);
    

    dim3 blockDim(8, 8, 8);
    dim3 gridDim((output_cols + blockDim.x - 1) / blockDim.x,
                 (output_rows + blockDim.y - 1) / blockDim.y,
                 (output_depth + blockDim.z - 1) / blockDim.z);
    

    conv3d_kernel<<<gridDim, blockDim>>>(d_input, d_kernel, d_output,
                                           input_depth, input_rows, input_cols,
                                           kernel_depth, kernel_rows, kernel_cols,
                                           output_depth, output_rows, output_cols);
    

    cudaDeviceSynchronize();
    

    cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}
