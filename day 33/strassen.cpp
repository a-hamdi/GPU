#include "matrix_mult.h"
#include <hip/hip_runtime.h>

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
__global__ void matrix_add_kernel(
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
__global__ void matrix_sub_kernel(
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

// Helper function to allocate device memory and copy data
void allocate_and_copy(float** d_ptr, const float* h_ptr, size_t size) {
    HIP_CHECK(hipMalloc(d_ptr, size));
    if (h_ptr) {
        HIP_CHECK(hipMemcpy(*d_ptr, h_ptr, size, hipMemcpyHostToDevice));
    }
}

// Helper function to perform matrix addition on device
void matrix_add(const float* A, const float* B, float* C, int N) {
    dim3 block(256);
    dim3 grid((N * N + block.x - 1) / block.x);
    matrix_add_kernel<<<grid, block>>>(A, B, C, N);
}

// Helper function to perform matrix subtraction on device
void matrix_sub(const float* A, const float* B, float* C, int N) {
    dim3 block(256);
    dim3 grid((N * N + block.x - 1) / block.x);
    matrix_sub_kernel<<<grid, block>>>(A, B, C, N);
}

// Recursive Strassen's algorithm implementation
void strassen_recursive(
    float* A, float* B, float* C,
    float* P1, float* P2, float* P3, float* P4, float* P5, float* P6, float* P7,
    float* temp1, float* temp2,
    int N)
{
    if (N <= 64) {  // Base case: use standard matrix multiplication for small matrices
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
        
        // Simple matrix multiplication kernel
        auto simple_gemm = [&]() {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < N; k++) {
                        sum += A[i * N + k] * B[k * N + j];
                    }
                    C[i * N + j] = sum;
                }
            }
        };
        
        simple_gemm();
        return;
    }

    int new_size = N / 2;
    size_t quarter_size = new_size * new_size * sizeof(float);

    // Allocate temporary matrices for the quadrants
    float *A11, *A12, *A21, *A22;
    float *B11, *B12, *B21, *B22;
    float *C11, *C12, *C21, *C22;
    
    allocate_and_copy(&A11, nullptr, quarter_size);
    allocate_and_copy(&A12, nullptr, quarter_size);
    allocate_and_copy(&A21, nullptr, quarter_size);
    allocate_and_copy(&A22, nullptr, quarter_size);
    allocate_and_copy(&B11, nullptr, quarter_size);
    allocate_and_copy(&B12, nullptr, quarter_size);
    allocate_and_copy(&B21, nullptr, quarter_size);
    allocate_and_copy(&B22, nullptr, quarter_size);
    allocate_and_copy(&C11, nullptr, quarter_size);
    allocate_and_copy(&C12, nullptr, quarter_size);
    allocate_and_copy(&C21, nullptr, quarter_size);
    allocate_and_copy(&C22, nullptr, quarter_size);

    // Split input matrices
    for (int i = 0; i < new_size; i++) {
        for (int j = 0; j < new_size; j++) {
            A11[i * new_size + j] = A[i * N + j];
            A12[i * new_size + j] = A[i * N + j + new_size];
            A21[i * new_size + j] = A[(i + new_size) * N + j];
            A22[i * new_size + j] = A[(i + new_size) * N + j + new_size];
            
            B11[i * new_size + j] = B[i * N + j];
            B12[i * new_size + j] = B[i * N + j + new_size];
            B21[i * new_size + j] = B[(i + new_size) * N + j];
            B22[i * new_size + j] = B[(i + new_size) * N + j + new_size];
        }
    }

    // Compute the seven products
    // P1 = (A11 + A22)(B11 + B22)
    matrix_add(A11, A22, temp1, new_size);
    matrix_add(B11, B22, temp2, new_size);
    strassen_recursive(temp1, temp2, P1, P1, P2, P3, P4, P5, P6, P7, temp1, temp2, new_size);

    // P2 = (A21 + A22)B11
    matrix_add(A21, A22, temp1, new_size);
    strassen_recursive(temp1, B11, P2, P1, P2, P3, P4, P5, P6, P7, temp1, temp2, new_size);

    // P3 = A11(B12 - B22)
    matrix_sub(B12, B22, temp1, new_size);
    strassen_recursive(A11, temp1, P3, P1, P2, P3, P4, P5, P6, P7, temp1, temp2, new_size);

    // P4 = A22(B21 - B11)
    matrix_sub(B21, B11, temp1, new_size);
    strassen_recursive(A22, temp1, P4, P1, P2, P3, P4, P5, P6, P7, temp1, temp2, new_size);

    // P5 = (A11 + A12)B22
    matrix_add(A11, A12, temp1, new_size);
    strassen_recursive(temp1, B22, P5, P1, P2, P3, P4, P5, P6, P7, temp1, temp2, new_size);

    // P6 = (A21 - A11)(B11 + B12)
    matrix_sub(A21, A11, temp1, new_size);
    matrix_add(B11, B12, temp2, new_size);
    strassen_recursive(temp1, temp2, P6, P1, P2, P3, P4, P5, P6, P7, temp1, temp2, new_size);

    // P7 = (A12 - A22)(B21 + B22)
    matrix_sub(A12, A22, temp1, new_size);
    matrix_add(B21, B22, temp2, new_size);
    strassen_recursive(temp1, temp2, P7, P1, P2, P3, P4, P5, P6, P7, temp1, temp2, new_size);

    // Calculate C11, C12, C21, C22
    // C11 = P1 + P4 - P5 + P7
    matrix_add(P1, P4, temp1, new_size);
    matrix_sub(P7, P5, temp2, new_size);
    matrix_add(temp1, temp2, C11, new_size);

    // C12 = P3 + P5
    matrix_add(P3, P5, C12, new_size);

    // C21 = P2 + P4
    matrix_add(P2, P4, C21, new_size);

    // C22 = P1 - P2 + P3 + P6
    matrix_sub(P1, P2, temp1, new_size);
    matrix_add(P3, P6, temp2, new_size);
    matrix_add(temp1, temp2, C22, new_size);

    // Combine results into C
    for (int i = 0; i < new_size; i++) {
        for (int j = 0; j < new_size; j++) {
            C[i * N + j] = C11[i * new_size + j];
            C[i * N + j + new_size] = C12[i * new_size + j];
            C[(i + new_size) * N + j] = C21[i * new_size + j];
            C[(i + new_size) * N + j + new_size] = C22[i * new_size + j];
        }
    }

    // Cleanup
    hipFree(A11); hipFree(A12); hipFree(A21); hipFree(A22);
    hipFree(B11); hipFree(B12); hipFree(B21); hipFree(B22);
    hipFree(C11); hipFree(C12); hipFree(C21); hipFree(C22);
}

PerfResult strassen_multiply(const float* A, const float* B, float* C, int N) {
    PerfResult result = {0.0, 0.0, "Strassen", 0.0};  // Initialize all fields

    // Ensure N is a power of 2
    if ((N & (N - 1)) != 0) {
        // Handle non-power-of-2 sizes by padding
        int new_size = 1;
        while (new_size < N) new_size *= 2;
        
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

        // Allocate device memory for padded matrices
        float *d_A, *d_B, *d_C;
        float *d_P1, *d_P2, *d_P3, *d_P4, *d_P5, *d_P6, *d_P7;
        float *d_temp1, *d_temp2;

        size_t size = new_size * new_size * sizeof(float);
        allocate_and_copy(&d_A, A_padded.data(), size);
        allocate_and_copy(&d_B, B_padded.data(), size);
        allocate_and_copy(&d_C, nullptr, size);
        allocate_and_copy(&d_P1, nullptr, size);
        allocate_and_copy(&d_P2, nullptr, size);
        allocate_and_copy(&d_P3, nullptr, size);
        allocate_and_copy(&d_P4, nullptr, size);
        allocate_and_copy(&d_P5, nullptr, size);
        allocate_and_copy(&d_P6, nullptr, size);
        allocate_and_copy(&d_P7, nullptr, size);
        allocate_and_copy(&d_temp1, nullptr, size);
        allocate_and_copy(&d_temp2, nullptr, size);

        // Time the Strassen multiplication
        hipEvent_t start, stop;
        hipEventCreate(&start);
        hipEventCreate(&stop);

        hipEventRecord(start);
        strassen_recursive(d_A, d_B, d_C, d_P1, d_P2, d_P3, d_P4, d_P5, d_P6, d_P7, d_temp1, d_temp2, new_size);
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

        // Calculate GFLOPS (approximate for Strassen's algorithm)
        double operations = 7.0 * pow(new_size, log2(7)) - 6.0 * new_size * new_size;
        result.gflops = (operations / (milliseconds * 1e-3)) / 1e9;

        // Cleanup
        hipFree(d_A); hipFree(d_B); hipFree(d_C);
        hipFree(d_P1); hipFree(d_P2); hipFree(d_P3); hipFree(d_P4);
        hipFree(d_P5); hipFree(d_P6); hipFree(d_P7);
        hipFree(d_temp1); hipFree(d_temp2);
        hipEventDestroy(start);
        hipEventDestroy(stop);
    } else {
        // For power-of-2 sizes, proceed directly
        float *d_A, *d_B, *d_C;
        float *d_P1, *d_P2, *d_P3, *d_P4, *d_P5, *d_P6, *d_P7;
        float *d_temp1, *d_temp2;

        size_t size = N * N * sizeof(float);
        allocate_and_copy(&d_A, A, size);
        allocate_and_copy(&d_B, B, size);
        allocate_and_copy(&d_C, nullptr, size);
        allocate_and_copy(&d_P1, nullptr, size);
        allocate_and_copy(&d_P2, nullptr, size);
        allocate_and_copy(&d_P3, nullptr, size);
        allocate_and_copy(&d_P4, nullptr, size);
        allocate_and_copy(&d_P5, nullptr, size);
        allocate_and_copy(&d_P6, nullptr, size);
        allocate_and_copy(&d_P7, nullptr, size);
        allocate_and_copy(&d_temp1, nullptr, size);
        allocate_and_copy(&d_temp2, nullptr, size);

        hipEvent_t start, stop;
        hipEventCreate(&start);
        hipEventCreate(&stop);

        hipEventRecord(start);
        strassen_recursive(d_A, d_B, d_C, d_P1, d_P2, d_P3, d_P4, d_P5, d_P6, d_P7, d_temp1, d_temp2, N);
        hipEventRecord(stop);
        hipEventSynchronize(stop);

        float milliseconds = 0;
        hipEventElapsedTime(&milliseconds, start, stop);
        result.time_ms = milliseconds;

        HIP_CHECK(hipMemcpy(C, d_C, size, hipMemcpyDeviceToHost));

        // Calculate GFLOPS (approximate for Strassen's algorithm)
        double operations = 7.0 * pow(N, log2(7)) - 6.0 * N * N;
        result.gflops = (operations / (milliseconds * 1e-3)) / 1e9;

        // Cleanup
        hipFree(d_A); hipFree(d_B); hipFree(d_C);
        hipFree(d_P1); hipFree(d_P2); hipFree(d_P3); hipFree(d_P4);
        hipFree(d_P5); hipFree(d_P6); hipFree(d_P7);
        hipFree(d_temp1); hipFree(d_temp2);
        hipEventDestroy(start);
        hipEventDestroy(stop);
    }

    return result;
} 