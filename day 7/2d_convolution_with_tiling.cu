#include <stdio.h>
#include <iostream>

// I'm assuming that the mask and the matrix to be square for simplicity
#define Mask_width 5
#define shared_size (32 + Mask_width - 1)
__constant__ float M[Mask_width][Mask_width];

__global__ void twod_convolution_kernel(const float* A, float* C, int n) {
    int threadx = threadIdx.x;
    int thready = threadIdx.y;
    int i = blockDim.x * blockIdx.x + threadx;
    int j = blockDim.y * blockIdx.y + thready;
    
    __shared__ float S_A[shared_size][shared_size];

    // Load main data
    if ((i < n) && (j < n)) {
        S_A[threadx + Mask_width/2][thready + Mask_width/2] = A[i*n+j];
    }

    // Load left halo
    if (threadx < Mask_width/2) {
        int left_idx = blockIdx.x * blockDim.x - (Mask_width/2) + threadx;
        if (left_idx >= 0 && j < n) {
            S_A[threadx][thready + Mask_width/2] = A[left_idx*n+j];
        }
        else {
            S_A[threadx][thready + Mask_width/2] = 0.0f;
        }
    }

    // Load right halo
    if (threadx < Mask_width/2) {
        int right_idx = blockIdx.x * blockDim.x + blockDim.x + threadx;
        if (right_idx < n && j < n) {
            S_A[threadx + blockDim.x + Mask_width/2][thready + Mask_width/2] = A[right_idx*n+j];
        }
        else {
            S_A[threadx + blockDim.x + Mask_width/2][thready + Mask_width/2] = 0.0f;
        }
    }

    // Load top halo
    if (thready < Mask_width/2) {
        int top_idy = j - (Mask_width/2) + thready;
        if (top_idy >= 0 && i < n) {
            S_A[threadx + Mask_width/2][thready] = A[i*n+top_idy];
        }
        else {
            S_A[threadx + Mask_width/2][thready] = 0.0f;
        }
    }

    // Load bottom halo
    if (thready < Mask_width/2) {
        int bottom_idy = j + blockDim.y + thready;
        if (bottom_idy < n && i < n) {
            S_A[threadx + Mask_width/2][thready + blockDim.y + Mask_width/2] = A[i*n+bottom_idy];
        }
        else {
            S_A[threadx + Mask_width/2][thready + blockDim.y + Mask_width/2] = 0.0f;
        }
    }

    __syncthreads();

    if ((i < n) && (j < n)) {
        float result = 0.0f;
        for (int k = 0; k < Mask_width; k++) {
            for (int x = 0; x < Mask_width; x++) {
                result += S_A[threadx + k][thready + x] * M[k][x];
            }
        }
        C[i*n+j] = result;
    }
}

void checkCudaError(const char* message) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "%s - CUDA Error: %s\n", message, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

int main() {
    int n = 10;
    float *h_A = (float*)malloc(n * n * sizeof(float));
    float *h_C = (float*)malloc(n * n * sizeof(float));
    float d_M[Mask_width][Mask_width];

    for (int i = 0; i < Mask_width; i++) {
        for (int j = 0; j < Mask_width; j++) {
            d_M[i][j] = 5;
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            h_A[i*n + j] = 3;
        }
    }

    float *d_a, *d_c;
    cudaMalloc(&d_a, n*n*sizeof(float));
    cudaMalloc(&d_c, n*n*sizeof(float));
    cudaMemcpy(d_a, h_A, n*n*sizeof(float), cudaMemcpyHostToDevice);
    checkCudaError("Failed to copy input data to device");
    cudaMemcpyToSymbol(M, d_M, Mask_width*Mask_width*sizeof(float));
    checkCudaError("Failed to copy mask data to device");

    dim3 dimBlock(32, 32);
    dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x, (n + dimBlock.y - 1) / dimBlock.y);
    twod_convolution_kernel<<<dimGrid, dimBlock>>>(d_a, d_c, n);
    checkCudaError("Failed to execute the kernel");
    
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_c, n*n*sizeof(float), cudaMemcpyDeviceToHost);
    checkCudaError("Failed to copy output data to host");

    // Print results
    printf("Results:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f ", h_C[i*n + j]);
        }
        printf("\n");
    }

    // Clean up
    cudaFree(d_a);
    cudaFree(d_c);
    free(h_A);
    free(h_C);

    return 0;
}

