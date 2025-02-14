#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <iostream>

#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error in %s:%d: %s\n", __FILE__, __LINE__, \
                hipGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define ROCBLAS_CHECK(call) do { \
    rocblas_status status = call; \
    if (status != rocblas_status_success) { \
        fprintf(stderr, "rocBLAS error in %s:%d: %d\n", __FILE__, __LINE__, \
                static_cast<int>(status)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

void printMatrix(const char* name, float* matrix, int rows, int cols) {
    std::cout << name << ":\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

int main() {
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    int M = 2, N = 3, K = 4;
    float *h_A, *h_B, *h_C;
    h_A = new float[M * K];
    h_B = new float[K * N];
    h_C = new float[M * N];

    // Initialize matrices
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            h_A[i * K + j] = i + j;
        }
    }

    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            h_B[i * N + j] = i + j;
        }
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, M * K * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_B, K * N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_C, M * N * sizeof(float)));

    // Copy matrices to device
    HIP_CHECK(hipMemcpy(d_A, h_A, M * K * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B, K * N * sizeof(float), hipMemcpyHostToDevice));

    // Set up and perform matrix multiplication: C = alpha*A*B + beta*C
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Note: rocBLAS uses column-major order, so we transpose the operation
    // C = A*B becomes C' = B'*A'
    ROCBLAS_CHECK(rocblas_sgemm(handle,
                               rocblas_operation_none,  // op(A) = A
                               rocblas_operation_none,  // op(B) = B
                               N, M, K,                 // m, n, k
                               &alpha,                  // alpha
                               d_B, N,                  // B, ldb
                               d_A, K,                  // A, lda
                               &beta,                   // beta
                               d_C, N));               // C, ldc

    // Copy result back to host
    HIP_CHECK(hipMemcpy(h_C, d_C, M * N * sizeof(float), hipMemcpyDeviceToHost));

    // Print matrices
    printMatrix("Matrix A", h_A, M, K);
    printMatrix("Matrix B", h_B, K, N);
    
    // Convert from column-major to row-major for printing
    float* h_C_row = new float[M * N];
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            h_C_row[i * N + j] = h_C[j * M + i];
        }
    }
    printMatrix("Matrix C = A * B", h_C_row, M, N);

    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_row;
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    ROCBLAS_CHECK(rocblas_destroy_handle(handle));

    return 0;
} 