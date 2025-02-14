#include <hip/hip_runtime.h>
#include <iostream>
#include <cmath>
#include <random>

__global__ void LayerNorm(const float* A, float* B, int rows, int cols) {
    // Calculate row index
    int row = blockIdx.x;

    if (row < rows) {
        extern __shared__ float shared[];
        float* row_data = &shared[0];
        float* temp_storage = &shared[cols];
        
        int tid = threadIdx.x;

        // Copy row data to shared memory
        for (int col = tid; col < cols; col += blockDim.x) {
            row_data[col] = A[row * cols + col];
        }
        __syncthreads();

        // Compute mean using parallel reduction
        float thread_sum = 0.0f;
        for (int col = tid; col < cols; col += blockDim.x) {
            thread_sum += row_data[col];
        }
        temp_storage[tid] = thread_sum;
        __syncthreads();
        
        // Reduce within block
        for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                temp_storage[tid] += temp_storage[tid + stride];
            }
            __syncthreads();
        }
        
        float mean = temp_storage[0] / cols;
        
        // Compute variance using parallel reduction
        thread_sum = 0.0f;
        for (int col = tid; col < cols; col += blockDim.x) {
            float diff = row_data[col] - mean;
            thread_sum += diff * diff;
        }
        temp_storage[tid] = thread_sum;
        __syncthreads();
        
        // Reduce within block
        for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                temp_storage[tid] += temp_storage[tid + stride];
            }
            __syncthreads();
        }
        
        float variance = temp_storage[0] / cols;
        float stddev = sqrtf(variance + 1e-5f);

        // Normalize
        for (int col = tid; col < cols; col += blockDim.x) {
            B[row * cols + col] = (row_data[col] - mean) / stddev;
        }
    }
}

int main() {
    const int rows = 10, cols = 10;
    float *A, *B;

    // Allocate host memory
    A = (float*)malloc(rows * cols * sizeof(float));
    B = (float*)malloc(rows * cols * sizeof(float));

    // Initialize input matrix with random values between 0 and 1
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            A[i * cols + j] = dis(gen);
        }
    }

    // Allocate device memory
    float *d_a, *d_b;
    hipMalloc(&d_a, rows * cols * sizeof(float));
    hipMalloc(&d_b, rows * cols * sizeof(float));

    // Copy data to device
    hipMemcpy(d_a, A, rows * cols * sizeof(float), hipMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 32;  // Reduced thread count since we have small matrices
    dim3 blockDim(threadsPerBlock);
    dim3 gridDim(rows);  // One block per row
    size_t shared_memory_size = (cols + threadsPerBlock) * sizeof(float);

    hipLaunchKernelGGL(LayerNorm,
                       gridDim,
                       blockDim,
                       shared_memory_size, 0,
                       d_a, d_b, rows, cols);
    
    // Synchronize device
    hipDeviceSynchronize();

    // Copy result back to host
    hipMemcpy(B, d_b, rows * cols * sizeof(float), hipMemcpyDeviceToHost);

    // Print results
    printf("Input Matrix A:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.4f ", A[i * cols + j]);
        }
        printf("\n");
    }

    printf("\nNormalized Matrix B:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.4f ", B[i * cols + j]);
        }
        printf("\n");
    }

    // Verify normalization (mean should be ~0, variance should be ~1)
    for (int i = 0; i < rows; i++) {
        float row_mean = 0.0f;
        float row_var = 0.0f;
        
        // Calculate mean
        for (int j = 0; j < cols; j++) {
            row_mean += B[i * cols + j];
        }
        row_mean /= cols;
        
        // Calculate variance
        for (int j = 0; j < cols; j++) {
            float diff = B[i * cols + j] - row_mean;
            row_var += diff * diff;
        }
        row_var /= cols;
        
        printf("Row %d: Mean = %.6f, Variance = %.6f\n", i, row_mean, row_var);
    }

    // Free memory
    hipFree(d_a);
    hipFree(d_b);
    free(A);
    free(B);

    return 0;
} 