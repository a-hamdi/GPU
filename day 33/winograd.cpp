#include "matrix_mult.h"
#include <hip/hip_runtime.h>
#include <cmath>

// Define HIP_CHECK macro for error handling
#define HIP_CHECK(call) do {                                                        \
    hipError_t err = call;                                                         \
    if (err != hipSuccess) {                                                       \
        printf("HIP error %s:%d: '%s'\n", __FILE__, __LINE__,                     \
               hipGetErrorString(err));                                            \
        exit(1);                                                                   \
    }                                                                              \
} while(0)

// Kernel for matrix addition
__global__ void winograd_add_kernel(
    const float* A,
    const float* B,
    float* C,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Kernel for matrix subtraction
__global__ void winograd_sub_kernel(
    const float* A,
    const float* B,
    float* C,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * N) {
        C[idx] = A[idx] - B[idx];
    }
}

// Kernel for computing intermediate matrices
__global__ void compute_intermediate_kernel(
    const float* A,
    const float* B,
    float* S,
    float* T,
    int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N/2 && col < N/2) {
        // Compute S = (A21 + A22) - A11
        int idx = row * N/2 + col;
        S[idx] = (A[(row + N/2) * N + col] + A[(row + N/2) * N + (col + N/2)]) - A[row * N + col];
        
        // Compute T = (B12 - B11) + B22
        T[idx] = (B[row * N + (col + N/2)] - B[row * N + col]) + B[(row + N/2) * N + (col + N/2)];
    }
}

// Main Winograd multiplication kernel for 2x2 block
__global__ void winograd_multiply_kernel(
    const float* A,
    const float* B,
    float* C,
    int N)
{
    __shared__ float As[32][32];  // Assuming block size <= 32
    __shared__ float Bs[32][32];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    
    float sum = 0.0f;
    
    for (int i = 0; i < N/32; ++i) {
        if (row < N && i * 32 + tx < N)
            As[ty][tx] = A[row * N + i * 32 + tx];
        else
            As[ty][tx] = 0.0f;
            
        if (i * 32 + ty < N && col < N)
            Bs[ty][tx] = B[(i * 32 + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;
            
        __syncthreads();
        
        for (int k = 0; k < 32; ++k)
            sum += As[ty][k] * Bs[k][tx];
            
        __syncthreads();
    }
    
    if (row < N && col < N)
        C[row * N + col] = sum;
}

PerfResult winograd_multiply(const float* A, const float* B, float* C, int N) {
    PerfResult result = {0.0, 0.0, "Winograd", 0.0};  // Initialize all fields

    // Ensure N is even
    if (N % 2 != 0) {
        // Handle odd sizes by padding
        int new_size = N + 1;
        
        // Create padded matrices
        std::vector<float> A_padded(new_size * new_size, 0.0f);
        std::vector<float> B_padded(new_size * new_size, 0.0f);
        std::vector<float> C_padded(new_size * new_size, 0.0f);

        // Copy original matrices to padded ones
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A_padded[i * new_size + j] = A[i * N + j];
                B_padded[i * new_size + j] = B[i * N + j];
            }
        }

        // Allocate device memory
        float *d_A, *d_B, *d_C;
        float *d_S, *d_T;  // Intermediate matrices
        float *d_M1, *d_M2, *d_M3, *d_M4, *d_M5, *d_M6, *d_M7;
        
        size_t size = new_size * new_size * sizeof(float);
        size_t half_size = (new_size/2) * (new_size/2) * sizeof(float);
        
        HIP_CHECK(hipMalloc(&d_A, size));
        HIP_CHECK(hipMalloc(&d_B, size));
        HIP_CHECK(hipMalloc(&d_C, size));
        HIP_CHECK(hipMalloc(&d_S, half_size));
        HIP_CHECK(hipMalloc(&d_T, half_size));
        HIP_CHECK(hipMalloc(&d_M1, half_size));
        HIP_CHECK(hipMalloc(&d_M2, half_size));
        HIP_CHECK(hipMalloc(&d_M3, half_size));
        HIP_CHECK(hipMalloc(&d_M4, half_size));
        HIP_CHECK(hipMalloc(&d_M5, half_size));
        HIP_CHECK(hipMalloc(&d_M6, half_size));
        HIP_CHECK(hipMalloc(&d_M7, half_size));

        // Copy data to device
        HIP_CHECK(hipMemcpy(d_A, A_padded.data(), size, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_B, B_padded.data(), size, hipMemcpyHostToDevice));

        // Set up timing
        hipEvent_t start, stop;
        hipEventCreate(&start);
        hipEventCreate(&stop);

        hipEventRecord(start);

        // Compute intermediate matrices S and T
        dim3 block(16, 16);
        dim3 grid((new_size/2 + block.x - 1)/block.x, (new_size/2 + block.y - 1)/block.y);
        compute_intermediate_kernel<<<grid, block>>>(d_A, d_B, d_S, d_T, new_size);

        // Main multiplication using Winograd's algorithm
        dim3 block_main(32, 32);
        dim3 grid_main((new_size + block_main.x - 1)/block_main.x, 
                      (new_size + block_main.y - 1)/block_main.y);
        winograd_multiply_kernel<<<grid_main, block_main>>>(d_A, d_B, d_C, new_size);

        hipEventRecord(stop);
        hipEventSynchronize(stop);

        float milliseconds = 0;
        hipEventElapsedTime(&milliseconds, start, stop);
        result.time_ms = milliseconds;

        // Copy result back and remove padding
        HIP_CHECK(hipMemcpy(C_padded.data(), d_C, size, hipMemcpyDeviceToHost));
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                C[i * N + j] = C_padded[i * new_size + j];
            }
        }

        // Calculate GFLOPS
        double operations = 2.0 * N * N * N;  // Approximate for Winograd
        result.gflops = (operations / (milliseconds * 1e-3)) / 1e9;

        // Cleanup
        hipFree(d_A);
        hipFree(d_B);
        hipFree(d_C);
        hipFree(d_S);
        hipFree(d_T);
        hipFree(d_M1);
        hipFree(d_M2);
        hipFree(d_M3);
        hipFree(d_M4);
        hipFree(d_M5);
        hipFree(d_M6);
        hipFree(d_M7);
        hipEventDestroy(start);
        hipEventDestroy(stop);
    } else {
        // For even sizes, proceed directly
        float *d_A, *d_B, *d_C;
        float *d_S, *d_T;
        float *d_M1, *d_M2, *d_M3, *d_M4, *d_M5, *d_M6, *d_M7;
        
        size_t size = N * N * sizeof(float);
        size_t half_size = (N/2) * (N/2) * sizeof(float);
        
        HIP_CHECK(hipMalloc(&d_A, size));
        HIP_CHECK(hipMalloc(&d_B, size));
        HIP_CHECK(hipMalloc(&d_C, size));
        HIP_CHECK(hipMalloc(&d_S, half_size));
        HIP_CHECK(hipMalloc(&d_T, half_size));
        HIP_CHECK(hipMalloc(&d_M1, half_size));
        HIP_CHECK(hipMalloc(&d_M2, half_size));
        HIP_CHECK(hipMalloc(&d_M3, half_size));
        HIP_CHECK(hipMalloc(&d_M4, half_size));
        HIP_CHECK(hipMalloc(&d_M5, half_size));
        HIP_CHECK(hipMalloc(&d_M6, half_size));
        HIP_CHECK(hipMalloc(&d_M7, half_size));

        HIP_CHECK(hipMemcpy(d_A, A, size, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_B, B, size, hipMemcpyHostToDevice));

        hipEvent_t start, stop;
        hipEventCreate(&start);
        hipEventCreate(&stop);

        hipEventRecord(start);

        // Compute intermediate matrices
        dim3 block(16, 16);
        dim3 grid((N/2 + block.x - 1)/block.x, (N/2 + block.y - 1)/block.y);
        compute_intermediate_kernel<<<grid, block>>>(d_A, d_B, d_S, d_T, N);

        // Main multiplication
        dim3 block_main(32, 32);
        dim3 grid_main((N + block_main.x - 1)/block_main.x, 
                      (N + block_main.y - 1)/block_main.y);
        winograd_multiply_kernel<<<grid_main, block_main>>>(d_A, d_B, d_C, N);

        hipEventRecord(stop);
        hipEventSynchronize(stop);

        float milliseconds = 0;
        hipEventElapsedTime(&milliseconds, start, stop);
        result.time_ms = milliseconds;

        HIP_CHECK(hipMemcpy(C, d_C, size, hipMemcpyDeviceToHost));

        // Calculate GFLOPS
        double operations = 2.0 * N * N * N;  // Approximate for Winograd
        result.gflops = (operations / (milliseconds * 1e-3)) / 1e9;

        // Cleanup
        hipFree(d_A);
        hipFree(d_B);
        hipFree(d_C);
        hipFree(d_S);
        hipFree(d_T);
        hipFree(d_M1);
        hipFree(d_M2);
        hipFree(d_M3);
        hipFree(d_M4);
        hipFree(d_M5);
        hipFree(d_M6);
        hipFree(d_M7);
        hipEventDestroy(start);
        hipEventDestroy(stop);
    }

    return result;
} 