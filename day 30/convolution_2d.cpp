#include <hip/hip_runtime.h>
#include <stdio.h>
#include <iostream>

// Assuming that the mask and the matrix to be square for simplicity
#define Mask_width 5
#define BLOCK_SIZE 16
#define TILE_SIZE (BLOCK_SIZE + Mask_width - 1)

// HIP equivalent of __constant__ memory
__constant__ float M[Mask_width][Mask_width];

__global__ void twod_convolution_kernel(const float* A, float* C, int n) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_o = blockIdx.y * BLOCK_SIZE + ty;
    int col_o = blockIdx.x * BLOCK_SIZE + tx;
    int radius = Mask_width/2;

    // Calculate the input tile coordinates
    int row_i = row_o - radius;
    int col_i = col_o - radius;

    // Load input tile to shared memory with boundary check
    if (row_i >= 0 && row_i < n && col_i >= 0 && col_i < n) {
        tile[ty][tx] = A[row_i * n + col_i];
    } else {
        tile[ty][tx] = 0.0f;
    }

    // Load additional elements for the halo regions
    if (ty < Mask_width-1) {
        int row_h = row_i + BLOCK_SIZE;
        if (row_h >= 0 && row_h < n && col_i >= 0 && col_i < n) {
            tile[ty + BLOCK_SIZE][tx] = A[row_h * n + col_i];
        } else {
            tile[ty + BLOCK_SIZE][tx] = 0.0f;
        }
    }
    if (tx < Mask_width-1) {
        int col_h = col_i + BLOCK_SIZE;
        if (row_i >= 0 && row_i < n && col_h >= 0 && col_h < n) {
            tile[ty][tx + BLOCK_SIZE] = A[row_i * n + col_h];
        } else {
            tile[ty][tx + BLOCK_SIZE] = 0.0f;
        }
    }
    if (ty < Mask_width-1 && tx < Mask_width-1) {
        int row_h = row_i + BLOCK_SIZE;
        int col_h = col_i + BLOCK_SIZE;
        if (row_h >= 0 && row_h < n && col_h >= 0 && col_h < n) {
            tile[ty + BLOCK_SIZE][tx + BLOCK_SIZE] = A[row_h * n + col_h];
        } else {
            tile[ty + BLOCK_SIZE][tx + BLOCK_SIZE] = 0.0f;
        }
    }

    __syncthreads();

    // Compute convolution if within bounds
    if (row_o < n && col_o < n) {
        float sum = 0.0f;
        for (int i = 0; i < Mask_width; i++) {
            for (int j = 0; j < Mask_width; j++) {
                sum += M[i][j] * tile[ty + i][tx + j];
            }
        }
        C[row_o * n + col_o] = sum;
    }
}

void checkHipError(const char* message) {
    hipError_t error = hipGetLastError();
    if (error != hipSuccess) {
        fprintf(stderr, "%s - HIP Error: %s\n", message, hipGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

int main() {
    int n = 10;
    float *h_A = (float*)malloc(n * n * sizeof(float));
    float *h_C = (float*)malloc(n * n * sizeof(float));
    float h_M[Mask_width][Mask_width];

    // Initialize convolution mask
    for (int i = 0; i < Mask_width; i++) {
        for (int j = 0; j < Mask_width; j++) {
            h_M[i][j] = 1.0f;  // Changed to 1.0f for easier verification
        }
    }

    // Initialize input matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            h_A[i*n + j] = 1.0f;  // Changed to 1.0f for easier verification
        }
    }

    // Print input matrix
    printf("Input Matrix:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f ", h_A[i*n + j]);
        }
        printf("\n");
    }

    printf("\nConvolution Mask:\n");
    for (int i = 0; i < Mask_width; i++) {
        for (int j = 0; j < Mask_width; j++) {
            printf("%.2f ", h_M[i][j]);
        }
        printf("\n");
    }

    float *d_a, *d_c;
    hipMalloc(&d_a, n*n*sizeof(float));
    hipMalloc(&d_c, n*n*sizeof(float));
    
    // Copy input matrix to device
    hipMemcpy(d_a, h_A, n*n*sizeof(float), hipMemcpyHostToDevice);
    checkHipError("Failed to copy input data to device");
    
    // Copy mask to constant memory
    hipMemcpyToSymbol(HIP_SYMBOL(M), h_M, Mask_width*Mask_width*sizeof(float));
    checkHipError("Failed to copy mask data to device");

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                 (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Launch kernel
    hipLaunchKernelGGL(twod_convolution_kernel,
                       dimGrid,
                       dimBlock,
                       0, 0,
                       d_a, d_c, n);
    
    checkHipError("Failed to execute the kernel");
    hipDeviceSynchronize();

    // Copy result back to host
    hipMemcpy(h_C, d_c, n*n*sizeof(float), hipMemcpyDeviceToHost);
    checkHipError("Failed to copy output data to host");

    // Print results
    printf("\nResults:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f ", h_C[i*n + j]);
        }
        printf("\n");
    }

    // Clean up
    hipFree(d_a);
    hipFree(d_c);
    free(h_A);
    free(h_C);

    return 0;
} 