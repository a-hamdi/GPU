#include "matrix_mult.h"
#include <hip/hip_runtime.h>
#include <chrono>

// Define HIP_CHECK macro for error handling
#define HIP_CHECK(call) do {                                                        \
    hipError_t err = call;                                                         \
    if (err != hipSuccess) {                                                       \
        printf("HIP error %s:%d: '%s'\n", __FILE__, __LINE__,                     \
               hipGetErrorString(err));                                            \
        exit(1);                                                                   \
    }                                                                              \
} while(0)

// CSR SpMM kernel
__global__ void spmm_csr_kernel(const float* values, const int* row_ptr, 
                               const int* col_indices, const float* B, float* C,
                               int M, int K, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (int i = row_ptr[row]; i < row_ptr[row + 1]; i++) {
                int k = col_indices[i];
                sum += values[i] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

// COO SpMM kernel
__global__ void spmm_coo_kernel(const float* values, const int* row_indices,
                               const int* col_indices, const float* B, float* C,
                               int nnz, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nnz) {
        int row = row_indices[idx];
        int col = col_indices[idx];
        float val = values[idx];
        
        for (int j = 0; j < M; j++) {
            atomicAdd(&C[row * M + j], val * B[col * M + j]);
        }
    }
}

// Block-CSR SpMM kernel
__global__ void spmm_block_csr_kernel(const float* values, const int* row_ptr,
                                     const int* col_indices, const float* B, float* C,
                                     int N, int K, int M, int block_size) {
    int row_block = blockIdx.x;
    int thread_id = threadIdx.x;
    
    __shared__ float shared_B[32][32];  // Assuming max block size of 32
    
    int row = row_block * block_size;
    if (row < N) {
        for (int col = 0; col < M; col += block_size) {
            // Load block of matrix B into shared memory
            if (col + thread_id < M && thread_id < block_size) {
                for (int k = row_ptr[row_block]; k < row_ptr[row_block + 1]; k++) {
                    int col_block = col_indices[k] * block_size;
                    shared_B[thread_id][0] = B[(col_block + thread_id) * M + col];
                }
            }
            __syncthreads();
            
            // Compute block multiplication
            if (thread_id < block_size && row + thread_id < N && col < M) {
                float sum = 0.0f;
                for (int k = row_ptr[row_block]; k < row_ptr[row_block + 1]; k++) {
                    int val_idx = k * block_size * block_size + thread_id;
                    sum += values[val_idx] * shared_B[thread_id][0];
                }
                C[(row + thread_id) * M + col] = sum;
            }
            __syncthreads();
        }
    }
}

PerfResult sparse_gemm_csr(const CSRMatrix& A, const float* B, float* C,
                          int M, int K, int N) {
    float *d_values, *d_B, *d_C;
    int *d_row_ptr, *d_col_indices;
    
    // Get number of non-zero elements
    int nnz = A.values.size();
    
    // Allocate device memory
    HIP_CHECK(hipMalloc(&d_values, A.values.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_row_ptr, A.row_ptr.size() * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_col_indices, A.col_indices.size() * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_B, K * N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_C, M * N * sizeof(float)));
    
    // Copy data to device
    HIP_CHECK(hipMemcpy(d_values, A.values.data(), A.values.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_row_ptr, A.row_ptr.data(), A.row_ptr.size() * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_col_indices, A.col_indices.data(), A.col_indices.size() * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, B, K * N * sizeof(float), hipMemcpyHostToDevice));
    
    // Initialize C with zeros
    HIP_CHECK(hipMemset(d_C, 0, M * N * sizeof(float)));
    
    // Set up timing
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    
    // Launch kernel
    dim3 block(256);
    dim3 grid((M + block.x - 1) / block.x);
    
    HIP_CHECK(hipEventRecord(start));
    hipLaunchKernelGGL(spmm_csr_kernel, grid, block, 0, nullptr,
                       d_values, d_row_ptr, d_col_indices, d_B, d_C, M, K, N);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float milliseconds = 0;
    HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));
    
    // Copy result back to host
    HIP_CHECK(hipMemcpy(C, d_C, M * N * sizeof(float), hipMemcpyDeviceToHost));
    
    // Calculate performance metrics
    double gflops = (2.0 * nnz * N) / (milliseconds * 1e6);
    
    // Cleanup
    HIP_CHECK(hipFree(d_values));
    HIP_CHECK(hipFree(d_row_ptr));
    HIP_CHECK(hipFree(d_col_indices));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    
    return {milliseconds, gflops, "CSR", 0.0};
}

PerfResult sparse_gemm_coo(const COOMatrix& A, const float* B, float* C,
                          int M, int K, int N) {
    float *d_values, *d_B, *d_C;
    int *d_row_indices, *d_col_indices;
    
    // Allocate device memory
    HIP_CHECK(hipMalloc(&d_values, A.values.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_row_indices, A.row_indices.size() * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_col_indices, A.col_indices.size() * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_B, K * N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_C, M * N * sizeof(float)));
    
    // Copy data to device
    HIP_CHECK(hipMemcpy(d_values, A.values.data(), A.values.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_row_indices, A.row_indices.data(), A.row_indices.size() * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_col_indices, A.col_indices.data(), A.col_indices.size() * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, B, K * N * sizeof(float), hipMemcpyHostToDevice));
    
    // Initialize C with zeros
    HIP_CHECK(hipMemset(d_C, 0, M * N * sizeof(float)));
    
    // Set up timing
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    
    // Launch kernel
    dim3 block(256);
    dim3 grid((A.nnz + block.x - 1) / block.x);
    
    HIP_CHECK(hipEventRecord(start));
    hipLaunchKernelGGL(spmm_coo_kernel, grid, block, 0, nullptr,
                       d_values, d_row_indices, d_col_indices, d_B, d_C, A.nnz, M);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float milliseconds = 0;
    HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));
    
    // Copy result back to host
    HIP_CHECK(hipMemcpy(C, d_C, M * N * sizeof(float), hipMemcpyDeviceToHost));
    
    // Calculate performance metrics
    double gflops = (2.0 * A.nnz * N) / (milliseconds * 1e6);
    
    // Cleanup
    HIP_CHECK(hipFree(d_values));
    HIP_CHECK(hipFree(d_row_indices));
    HIP_CHECK(hipFree(d_col_indices));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    
    return {milliseconds, gflops, "COO", 0.0};
}

PerfResult sparse_gemm_block_csr(const BlockCSRMatrix& A, const float* B, float* C,
                                int M, int K, int N) {
    float *d_values, *d_B, *d_C;
    int *d_row_ptr, *d_col_indices;
    
    // Calculate nnz for Block-CSR
    int nnz = A.values.size() / (A.block_size * A.block_size);
    
    // Allocate device memory
    HIP_CHECK(hipMalloc(&d_values, A.values.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_row_ptr, A.row_ptr.size() * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_col_indices, A.col_indices.size() * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_B, K * N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_C, M * N * sizeof(float)));
    
    // Copy data to device
    HIP_CHECK(hipMemcpy(d_values, A.values.data(), A.values.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_row_ptr, A.row_ptr.data(), A.row_ptr.size() * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_col_indices, A.col_indices.data(), A.col_indices.size() * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, B, K * N * sizeof(float), hipMemcpyHostToDevice));
    
    // Initialize C with zeros
    HIP_CHECK(hipMemset(d_C, 0, M * N * sizeof(float)));
    
    // Set up timing
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    
    // Launch kernel
    dim3 block(A.block_size * A.block_size);
    dim3 grid((N + A.block_size - 1) / A.block_size);
    
    HIP_CHECK(hipEventRecord(start));
    hipLaunchKernelGGL(spmm_block_csr_kernel, grid, block, 0, nullptr,
                       d_values, d_row_ptr, d_col_indices, d_B, d_C,
                       N, K, M, A.block_size);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float milliseconds = 0;
    HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));
    
    // Copy result back to host
    HIP_CHECK(hipMemcpy(C, d_C, M * N * sizeof(float), hipMemcpyDeviceToHost));
    
    // Calculate performance metrics
    double gflops = (2.0 * nnz * A.block_size * A.block_size * N) / (milliseconds * 1e6);
    
    // Cleanup
    HIP_CHECK(hipFree(d_values));
    HIP_CHECK(hipFree(d_row_ptr));
    HIP_CHECK(hipFree(d_col_indices));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    
    return {milliseconds, gflops, "Block-CSR", 0.0};
} 