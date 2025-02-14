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

int main() {
    const int N = 10;
    float A[N], B[N], C[N];

    // Initialize input vectors
    for(int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = i;
    }

    // Create rocBLAS handle
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    // Allocate device memory
    float *d_a, *d_b;
    HIP_CHECK(hipMalloc(&d_a, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_b, N * sizeof(float)));

    // Copy data from host to device
    HIP_CHECK(hipMemcpy(d_a, A, N * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, B, N * sizeof(float), hipMemcpyHostToDevice));

    // Scaling factor
    const float alpha = 1.0f;

    // Perform vector addition: B = alpha*A + B
    ROCBLAS_CHECK(rocblas_saxpy(handle, N, &alpha, d_a, 1, d_b, 1));

    // Copy result back to host (result is in d_b)
    HIP_CHECK(hipMemcpy(C, d_b, N * sizeof(float), hipMemcpyDeviceToHost));

    // Print results
    std::cout << "Result vector: ";
    for(int i = 0; i < N; i++) {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;

    // Cleanup
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    ROCBLAS_CHECK(rocblas_destroy_handle(handle));

    return 0;
} 