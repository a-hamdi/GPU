#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cstdlib>
#define CUDA_MAX_NUM_THREADS 1024
#define BLOCK_SIZE 256

// CUDA Kernel for computing dL/dW
template <typename T>
__global__ void compute_dLdW(T* dLdY, T* input_unrolled, T* dLdW, int output_height, int output_width, int num_filters, int filter_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < filter_size && col < num_filters) {
        T sum = 0;
        for (int i = 0; i < output_height * output_width; i++) {
            sum += input_unrolled[i * filter_size + row] * dLdY[i * num_filters + col];
        }
        dLdW[row * num_filters + col] = sum;
    }
}

// CUDA Kernel for computing dL/dX
template <typename T>
__global__ void compute_dLdX(T* dLdY, T* weights, T* dLdX_unrolled, int output_height, int output_width, int num_filters, int filter_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < output_height * output_width && col < filter_size) {
        T sum = 0;
        for (int i = 0; i < num_filters; i++) {
            sum += dLdY[row * num_filters + i] * weights[col * num_filters + i];
        }
        dLdX_unrolled[row * filter_size + col] = sum;
    }
}
template <typename T>
__global__ void maxPoolingBackwardKernel(T* dLdY, T* input, T* dLdX, int input_height, int input_width, int pool_size, int stride) {
    int output_height = (input_height - pool_size) / stride + 1;
    int output_width = (input_width - pool_size) / stride + 1;
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < output_height && col < output_width) {
        T max_value = -INFINITY;
        int max_i = -1, max_j = -1;
        for (int i = 0; i < pool_size; i++) {
            for (int j = 0; j < pool_size; j++) {
                int input_row = row * stride + i;
                int input_col = col * stride + j;

                // Access input correctly, avoid out-of-bounds access
                if (input_row < input_height && input_col < input_width) {
                    if (input[input_row * input_width + input_col] > max_value) {
                        max_value = input[input_row * input_width + input_col];
                        max_i = input_row;
                        max_j = input_col;
                    }
                }
            }
        }

        // Ensure max_i and max_j are valid before accessing dLdX
        if (max_i != -1 && max_j != -1) {
            atomicAdd(&dLdX[max_i * input_width + max_j], dLdY[row * output_width + col]);
        }
    }
}

// Updated kernel signatures to match the calling convention
__global__ void unrollKernel(const float* input, float* input_unrolled,
                            const int input_channels, const int input_height, const int input_width,
                            const int kernel_size, const int output_height, const int output_width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = output_height * output_width;
    
    if (idx < total_elements) {
        int out_y = idx / output_width;
        int out_x = idx % output_width;
        
        for (int c = 0; c < input_channels; c++) {
            for (int ky = 0; ky < kernel_size; ky++) {
                for (int kx = 0; kx < kernel_size; kx++) {
                    int in_y = out_y + ky;
                    int in_x = out_x + kx;
                    
                    int unroll_idx = idx * (input_channels * kernel_size * kernel_size) +
                                   (c * kernel_size * kernel_size + ky * kernel_size + kx);
                    
                    int input_idx = c * (input_height * input_width) +
                                  in_y * input_width + in_x;
                    
                    input_unrolled[unroll_idx] = input[input_idx];
                }
            }
        }
    }
}


// Host function to launch Unrolling Kernel
void unrollInput(int input_channels, int input_height, int input_width, 
                int kernel_size, float* input, float* input_unrolled) {
    int output_height = input_height - kernel_size + 1;
    int output_width = input_width - kernel_size + 1;
    int total_output_elements = output_height * output_width;
    
    int threadsPerBlock = 256;
    int numBlocks = (total_output_elements + threadsPerBlock - 1) / threadsPerBlock;
    
    unrollKernel<<<numBlocks, threadsPerBlock>>>(
        input,                  // const float* input
        input_unrolled,        // float* input_unrolled
        input_channels,        // const int input_channels
        input_height,          // const int input_height
        input_width,           // const int input_width
        kernel_size,           // const int kernel_size
        output_height,         // const int output_height
        output_width          // const int output_width
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error in unroll: %s\n", cudaGetErrorString(error));
    }
    
    cudaDeviceSynchronize();
}

void convolutionBackward(int batch_size, int num_filters, int input_channels, int input_height, int input_width, int kernel_size, float* dLdY, float* input, float* weights, float* dLdX, float* dLdW) {
    int output_height = input_height - kernel_size + 1;
    int output_width = input_width - kernel_size + 1;
    int filter_size = input_channels * kernel_size * kernel_size;
    
    float* input_unrolled;
    float* dLdX_unrolled;
    cudaMalloc(&input_unrolled, output_height * output_width * filter_size * sizeof(float));
    cudaMalloc(&dLdX_unrolled, output_height * output_width * filter_size * sizeof(float));
    
    for (int n = 0; n < batch_size; n++) {
        unrollInput(input_channels, input_height, input_width, kernel_size, input + n * input_channels * input_height * input_width, input_unrolled);
        
        dim3 blockSize(16, 16);
        dim3 gridSize((output_width + blockSize.x - 1) / blockSize.x, (output_height + blockSize.y - 1) / blockSize.y);
        
        compute_dLdW<<<gridSize, blockSize>>>(dLdY, input_unrolled, dLdW, output_height, output_width, num_filters, filter_size);
        compute_dLdX<<<gridSize, blockSize>>>(dLdY, weights, dLdX_unrolled, output_height, output_width, num_filters, filter_size);
        cudaDeviceSynchronize();
    }
    
    cudaFree(input_unrolled);
    cudaFree(dLdX_unrolled);
}




// CUDA Kernel for Max Pooling
__global__ void maxPoolingKernel(float* input, float* output, int input_height, int input_width, int pool_size, int stride) {
    int output_height = (input_height - pool_size) / stride + 1;
    int output_width = (input_width - pool_size) / stride + 1;
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < output_height && col < output_width) {
        float max_value = -INFINITY;
        for (int i = 0; i < pool_size; i++) {
            for (int j = 0; j < pool_size; j++) {
                int input_row = row * stride + i;
                int input_col = col * stride + j;
                max_value = fmaxf(max_value, input[input_row * input_width + input_col]);
            }
        }
        output[row * output_width + col] = max_value;
    }
}

// CUDA Kernel for Matrix Multiplication (GEMM for Convolution)
__global__ void matrixMultiplicationKernel(float* input_unrolled, float* weights, float* output, 
                                         int output_height, int output_width, int num_filters, int filter_size) {
    // Calculate actual position
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_output_elements = output_height * output_width;
    
    if (idx < total_output_elements * num_filters) {
        int output_idx = idx / num_filters;  // Position in output feature map
        int filter_idx = idx % num_filters;  // Which filter we're using
        
        float sum = 0.0f;
        // Multiply unrolled input with the corresponding filter
        for (int i = 0; i < filter_size; i++) {
            sum += input_unrolled[output_idx * filter_size + i] * weights[i * num_filters + filter_idx];
        }
        output[idx] = sum;
    }
}




__global__ void convolutionKernel(const float* input_unrolled, const float* weights, float* output,
                                 const int output_size, const int num_filters, const int filter_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < output_size * num_filters) {
        int output_idx = idx / num_filters;
        int filter_idx = idx % num_filters;
        
        float sum = 0.0f;
        for (int i = 0; i < filter_size; i++) {
            sum += input_unrolled[output_idx * filter_size + i] * 
                   weights[i * num_filters + filter_idx];
        }
        output[idx] = sum;
    }
}

void convolutionForward(float* input, float* weights, float* output,
                       int batch_size, int num_filters, int input_channels,
                       int input_height, int input_width, int kernel_size) {
    int output_height = input_height - kernel_size + 1;
    int output_width = input_width - kernel_size + 1;
    int output_size = output_height * output_width;
    int filter_size = input_channels * kernel_size * kernel_size;
    
    // Allocate unrolled input matrix
    float* input_unrolled;
    size_t unrolled_size = output_size * filter_size * sizeof(float);
    cudaMalloc(&input_unrolled, unrolled_size);
    
    // Calculate grid and block dimensions
    int unroll_blocks = (output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int conv_blocks = (output_size * num_filters + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (int n = 0; n < batch_size; n++) {
        float* input_n = input + n * input_channels * input_height * input_width;
        float* output_n = output + n * num_filters * output_height * output_width;
        
        // Launch unroll kernel with correct parameters
        unrollKernel<<<unroll_blocks, BLOCK_SIZE>>>(
            input_n,
            input_unrolled,
            input_channels,
            input_height,
            input_width,
            kernel_size,
            output_height,
            output_width
        );
        
        // Check for kernel launch errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("Unroll kernel error: %s\n", cudaGetErrorString(error));
        }
        
        // Launch convolution kernel
        convolutionKernel<<<conv_blocks, BLOCK_SIZE>>>(
            input_unrolled,
            weights,
            output_n,
            output_size,
            num_filters,
            filter_size
        );
        
        // Check for kernel launch errors
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("Convolution kernel error: %s\n", cudaGetErrorString(error));
        }
        
        cudaDeviceSynchronize();
    }
    
    cudaFree(input_unrolled);
}
/*
void testConvNet() {
    // Test dimensions
    const int batch_size = 1;
    const int input_channels = 1;
    const int input_height = 4;
    const int input_width = 4;
    const int kernel_size = 3;
    const int num_filters = 2;
    
    // Calculate output dimensions
    const int output_height = input_height - kernel_size + 1;
    const int output_width = input_width - kernel_size + 1;
    
    // Allocate and initialize host memory
    float input[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    
    float weights[] = {
        1, 0, -1,
        1, 0, -1,
        1, 0, -1,
        // Second filter
        0, 1, -1,
        0, 1, -1,
        0, 1, -1
    };
    
    // Allocate device memory
    float *d_input, *d_weights, *d_output;
    
    size_t input_size = batch_size * input_channels * input_height * input_width * sizeof(float);
    size_t weights_size = num_filters * input_channels * kernel_size * kernel_size * sizeof(float);
    size_t output_size = batch_size * num_filters * output_height * output_width * sizeof(float);
    
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_weights, weights_size);
    cudaMalloc(&d_output, output_size);
    cudaMemset(d_output, 0, output_size);  // Initialize output to zero
    
    // Copy data to device
    cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, weights_size, cudaMemcpyHostToDevice);
    
    // Forward pass
    convolutionForward(d_input, d_weights, d_output,
                      batch_size, num_filters, input_channels,
                      input_height, input_width, kernel_size);
    
    // Copy results back to host
    float* output = new float[output_size/sizeof(float)];
    cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost);
    
    // Print results
    std::cout << "Forward Output:\n";
    for (int f = 0; f < num_filters; f++) {
        std::cout << "Filter " << f << ":\n";
        for (int i = 0; i < output_height; i++) {
            for (int j = 0; j < output_width; j++) {
                std::cout << output[f * output_height * output_width + i * output_width + j] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    
    // Cleanup
    delete[] output;
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_output);
}
*/
