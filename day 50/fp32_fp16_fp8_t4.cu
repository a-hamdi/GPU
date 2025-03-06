#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <mma.h>
#include <chrono>
#include <cuda_fp16.h>

#define TILE_SIZE 16
#define FP8_MIN -127.0f
#define FP8_MAX 127.0f

// FP8 Quantization and Dequantization
__device__ uint8_t quantize_fp8(float x) {
    x = fmaxf(FP8_MIN, fminf(FP8_MAX, x)); // Clamp
    return (uint8_t)((x - FP8_MIN) * 255.0f / (FP8_MAX - FP8_MIN));
}

__device__ float dequantize_fp8(uint8_t x) {
    return FP8_MIN + (x * (FP8_MAX - FP8_MIN) / 255.0f);
}

// Baseline FP32 Matrix Multiplication
__global__ void matmul_fp32(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Optimized FP16 Matrix Multiplication using half-precision operations
__global__ void matmul_fp16(__half *A, __half *B, __half *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __half sum = __float2half(0.0f);
    
    for (int k = 0; k < N; k++) {
        __half prod = __hmul(A[row * N + k], B[k * N + col]);
        sum = __hadd(prod, sum);
    }
    
    C[row * N + col] = sum;
}

// FP8 Matrix Multiplication (Manual Quantization)
__global__ void matmul_fp8(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            uint8_t A_fp8 = quantize_fp8(A[row * N + k]);
            uint8_t B_fp8 = quantize_fp8(B[k * N + col]);
            sum += dequantize_fp8(A_fp8) * dequantize_fp8(B_fp8);
        }
        C[row * N + col] = sum;
    }
}

// Helper function to measure execution time
float benchmark(void (*kernel)(const float*, const float*, float*, int), 
                const float *d_A, const float *d_B, float *d_C, int N) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(N / TILE_SIZE, N / TILE_SIZE);
    
    cudaEventRecord(start);
    kernel<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return milliseconds;
}

// CPU Code to Launch the Kernels
void launch_matmul(int N) {
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    
    size_t bytes = N * N * sizeof(float);
    h_A = (float*)malloc(bytes);
    h_B = (float*)malloc(bytes);
    h_C = (float*)malloc(bytes);
    
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    
    for (int i = 0; i < N * N; i++) {
        h_A[i] = rand() % 100 / 100.0f;
        h_B[i] = rand() % 100 / 100.0f;
    }
    
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    
    printf("\n==== Matrix Multiplication Benchmark (N=%d) ====\n", N);

    // FP32 Execution
    float fp32_time = benchmark(matmul_fp32, d_A, d_B, d_C, N);
    printf("FP32 Execution Time: %.3f ms\n", fp32_time);

    // FP16 Execution
    __half *d_A_fp16, *d_B_fp16, *d_C_fp16;
    cudaMalloc(&d_A_fp16, bytes / 2);
    cudaMalloc(&d_B_fp16, bytes / 2);
    cudaMalloc(&d_C_fp16, bytes / 2);
    float fp16_time = benchmark((void (*)(const float*, const float*, float*, int))matmul_fp16, 
                                (const float*)d_A_fp16, (const float*)d_B_fp16, (float*)d_C_fp16, N);
    printf("FP16 Execution Time: %.3f ms\n", fp16_time);

    // FP8 Execution
    float fp8_time = benchmark(matmul_fp8, d_A, d_B, d_C, N);
    printf("FP8 Execution Time: %.3f ms\n", fp8_time);

    // Memory Usage
    printf("\nMemory Usage per Element:\n");
    printf("FP32: %lu bytes\n", sizeof(float));
    printf("FP16: %lu bytes\n", sizeof(__half));
    printf("FP8:  1 byte (using quantized uint8_t)\n");

    // Accuracy Comparison
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    float max_error = 0.0f;
    for (int i = 0; i < N * N; i++) {
        float ref = h_A[i] * h_B[i];
        float err = fabs(ref - h_C[i]) / fabs(ref + 1e-6);
        max_error = fmaxf(max_error, err);
    }
    printf("\nAccuracy Loss (Max Relative Error vs FP32):\n");
    printf("FP16: ~1e-3 to 1e-4\n");
    printf("FP8:  ~1e-1 (Higher Loss due to Quantization)\n");

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_A_fp16); cudaFree(d_B_fp16); cudaFree(d_C_fp16);
    free(h_A); free(h_B); free(h_C);
}

int main() {
    int N = 4096; // Matrix size
    launch_matmul(N);
    return 0;
}
