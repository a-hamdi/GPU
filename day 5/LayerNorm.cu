#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

__global__ void LayerNorm(const float* A, float* B, int rows, int cols) {
    // Calculate row index
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows) {
        // Use shared memory for row-wise computation
        extern __shared__ float shared[];
        float* row_data = shared;

        // Copy row data to shared memory
        for (int col = threadIdx.y; col < cols; col += blockDim.y) {
            row_data[col] = A[row * cols + col];
        }
        __syncthreads();

        // Compute mean
        float mean = 0.0f;
        for (int col = 0; col < cols; col++) {
            mean += row_data[col];
        }
        mean /= cols;

        // Compute variance
        float variance = 0.0f;
        for (int col = 0; col < cols; col++) {
            variance += (row_data[col] - mean) * (row_data[col] - mean);
        }
        variance /= cols;
        float stddev = sqrtf(variance + 1e-7);

        // Normalize
        for (int col = threadIdx.y; col < cols; col += blockDim.y) {
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

    // Initialize input matrix
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            A[i * cols + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    // Allocate device memory
    float *d_a, *d_b;
    cudaMalloc(&d_a, rows * cols * sizeof(float));
    cudaMalloc(&d_b, rows * cols * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_a, A, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int blocksize = 256;
    int gridsize = (rows + blocksize - 1) / blocksize;
    size_t shared_memory_size = cols * sizeof(float);
    LayerNorm<<<gridsize, blocksize, shared_memory_size>>>(d_a, d_b, rows, cols);

    // Synchronize device
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(B, d_b, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results
    printf("A:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f ", A[i * cols + j]);
        }
        printf("\n");
    }

    printf("\nB:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f ", B[i * cols + j]);
        }
        printf("\n");
    }

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    free(A);
    free(B);

    return 0;
}