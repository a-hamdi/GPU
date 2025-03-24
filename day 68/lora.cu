#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>


#define CUDA_CHECK(error) { \
    if(error != cudaSuccess){ \
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

__global__ void lora_kernel(const float* x, const float* W, const float* A, const float* B, float* y,
                            int M, int N, int K, int R) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row < M && col < N) {
        float acc = 0.0f;
        
        for (int k = 0; k < K; ++k) {
            float sum_ab = 0.0f;
           
            for (int r = 0; r < R; ++r) {
                sum_ab += A[k * R + r] * B[r * N + col];
            }
           
            float w_eff = W[k * N + col] + sum_ab;
            acc += x[row * K + k] * w_eff;
        }
        y[row * N + col] = acc;
    }
}


void cpu_lora(const float* x, const float* W, const float* A, const float* B, float* y,
              int M, int N, int K, int R) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                float sum_ab = 0.0f;
                for (int r = 0; r < R; r++) {
                    sum_ab += A[k * R + r] * B[r * N + j];
                }
                float w_eff = W[k * N + j] + sum_ab;
                acc += x[i * K + k] * w_eff;
            }
            y[i * N + j] = acc;
        }
    }
}


void print_matrix(const float* mat, int M, int N, int max_rows = 5, int max_cols = 5) {
    for (int i = 0; i < std::min(M, max_rows); ++i) {
        for (int j = 0; j < std::min(N, max_cols); ++j) {
            std::cout << mat[i * N + j] << "\t";
        }
        std::cout << std::endl;
    }
}

int main() {
    
    int M = 128, K = 256, N = 64, R = 32;
    
    size_t size_x = M * K * sizeof(float);
    size_t size_W = K * N * sizeof(float);
    size_t size_A = K * R * sizeof(float);
    size_t size_B = R * N * sizeof(float);
    size_t size_y = M * N * sizeof(float);


    float *h_x = (float*)malloc(size_x);
    float *h_W = (float*)malloc(size_W);
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_y = (float*)malloc(size_y);
    float *h_y_cpu = (float*)malloc(size_y);


    for (int i = 0; i < M * K; i++) h_x[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; i++) h_W[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * R; i++) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < R * N; i++) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

  
    float *d_x, *d_W, *d_A, *d_B, *d_y;
    CUDA_CHECK(cudaMalloc((void**)&d_x, size_x));
    CUDA_CHECK(cudaMalloc((void**)&d_W, size_W));
    CUDA_CHECK(cudaMalloc((void**)&d_A, size_A));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size_B));
    CUDA_CHECK(cudaMalloc((void**)&d_y, size_y));

    CUDA_CHECK(cudaMemcpy(d_x, h_x, size_x, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W, h_W, size_W, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);

 
    CUDA_CHECK(cudaEventRecord(start, 0));


    lora_kernel<<<blocks, threads>>>(d_x, d_W, d_A, d_B, d_y, M, N, K, R);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

   
    float gpuTime = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));

   
    CUDA_CHECK(cudaMemcpy(h_y, d_y, size_y, cudaMemcpyDeviceToHost));

    
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_lora(h_x, h_W, h_A, h_B, h_y_cpu, M, N, K, R);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuTime = cpu_end - cpu_start;

   
    float max_diff = 0.0f;
    for (int i = 0; i < M * N; i++){
        float diff = fabs(h_y_cpu[i] - h_y[i]);
        if (diff > max_diff)
            max_diff = diff;
    }
    std::cout << "Max difference between CPU and GPU results: " << max_diff << std::endl;
    std::cout << "GPU kernel execution time: " << gpuTime << " ms" << std::endl;
    std::cout << "CPU execution time: " << cpuTime.count() << " ms" << std::endl;


    std::cout << "\nSample output from GPU result (first 5 rows, 5 cols):\n";
    print_matrix(h_y, M, N, 5, 5);

    std::cout << "\nSample output from CPU result (first 5 rows, 5 cols):\n";
    print_matrix(h_y_cpu, M, N, 5, 5);


    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    free(h_x); free(h_W); free(h_A); free(h_B); free(h_y); free(h_y_cpu);
    cudaFree(d_x); cudaFree(d_W); cudaFree(d_A); cudaFree(d_B); cudaFree(d_y);

    return 0;
}
