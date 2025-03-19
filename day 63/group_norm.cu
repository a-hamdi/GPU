#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(err) \
    if(err != cudaSuccess) {\
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;\
        exit(EXIT_FAILURE);\
    }

// Kernel that performs group normalization forward pass using parallel reduction.
__global__ void groupNormForward(const float *x, float *y, 
                                 const float *gamma, const float *beta,
                                 int N, int C, int H, int W, int G, float eps) {
    // Each block handles one sample (n) and one group (g)
    int n = blockIdx.x;
    int g = blockIdx.y;
    
    int group_channels = C / G;
    int spatial_size = H * W;
    int group_size = group_channels * spatial_size;
    int start_c = g * group_channels;
    
    // Use threadIdx.x to index within the group.
    int tid = threadIdx.x;
    
    // Each thread may process multiple elements.
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (int i = tid; i < group_size; i += blockDim.x) {
        int c_offset = i / spatial_size;   // index within the group channels
        int s = i % spatial_size;            // spatial index
        int channel = start_c + c_offset;
        int idx = n * C * spatial_size + channel * spatial_size + s;
        float val = x[idx];
        local_sum += val;
        local_sum_sq += val * val;
    }
    
    // Allocate shared memory for reduction (two arrays in one block).
    extern __shared__ float shared[];
    float* s_sum = shared;              // size: blockDim.x
    float* s_sum_sq = &shared[blockDim.x]; // size: blockDim.x

    s_sum[tid] = local_sum;
    s_sum_sq[tid] = local_sum_sq;
    __syncthreads();
    
    // Parallel reduction for sum and sum of squares.
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sum_sq[tid] += s_sum_sq[tid + stride];
        }
        __syncthreads();
    }
    
    // Compute mean and variance (only thread 0 does it, then broadcast).
    float mean = s_sum[0] / group_size;
    float var = s_sum_sq[0] / group_size - mean * mean;
    float inv_std = rsqrtf(var + eps);
    __syncthreads(); // ensure all threads see the computed mean and var

    // Normalize each element in the group.
    for (int i = tid; i < group_size; i += blockDim.x) {
        int c_offset = i / spatial_size;
        int s = i % spatial_size;
        int channel = start_c + c_offset;
        int idx = n * C * spatial_size + channel * spatial_size + s;
        float val = x[idx];
        float norm = (val - mean) * inv_std;
        y[idx] = gamma[channel] * norm + beta[channel];
    }
}

int main() {
    // Tensor dimensions.
    int N = 1;   // Batch size.
    int C = 4;   // Number of channels.
    int H = 2;   // Height.
    int W = 2;   // Width.
    int G = 2;   // Number of groups (C must be divisible by G).
    float eps = 1e-5;
    int tensor_size = N * C * H * W;
    
    // Host memory allocation.
    float h_x[tensor_size];
    float h_y[tensor_size];
    float h_gamma[C];
    float h_beta[C];
    
    // Initialize input tensor and parameters.
    for (int i = 0; i < tensor_size; i++) {
        h_x[i] = static_cast<float>(i);
    }
    for (int i = 0; i < C; i++) {
        h_gamma[i] = 1.0f;  // scaling factors.
        h_beta[i] = 0.0f;   // shifting factors.
    }
    
    // Device memory allocation.
    float *d_x, *d_y, *d_gamma, *d_beta;
    CUDA_CHECK(cudaMalloc(&d_x, tensor_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, tensor_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta, C * sizeof(float)));
    
    // Copy data to device.
    CUDA_CHECK(cudaMemcpy(d_x, h_x, tensor_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma, C * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta, C * sizeof(float), cudaMemcpyHostToDevice));
    
    // Set grid and block dimensions.
    int group_channels = C / G;
    int spatial_size = H * W;
    int group_size = group_channels * spatial_size;
    int threadsPerBlock = (group_size < 128) ? group_size : 128;
    // Shared memory size: two arrays of threadsPerBlock floats.
    size_t sharedMemSize = threadsPerBlock * 2 * sizeof(float);
    dim3 grid(N, G);
    
    // Launch the kernel.
    groupNormForward<<<grid, threadsPerBlock, sharedMemSize>>>(d_x, d_y, d_gamma, d_beta,
                                                                N, C, H, W, G, eps);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back to host.
    CUDA_CHECK(cudaMemcpy(h_y, d_y, tensor_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Print the normalized output.
    std::cout << "Group Norm Forward Output:" << std::endl;
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    int idx = n * C * H * W + c * H * W + h * W + w;
                    std::cout << h_y[idx] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }
    
    // Free device memory.
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));
    
    return 0;
}