#include <hip/hip_runtime.h>
#include <iostream>
#include <cmath>

#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error in %s:%d: %s\n", __FILE__, __LINE__, \
                hipGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void gelu_kernel(float* data, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        data[i] = 0.5f * data[i] * (1.0f + erff(data[i] / sqrtf(2.0f)));
    }
}

int main() {
    const int N = 1000000;
    float* A = new float[N];

    // Initialize array with values
    for (int i = 0; i < N; i++) {
        A[i] = -1.0f * (float)i / 2.0f;
    }

    // Print first 10 elements before GELU
    std::cout << "Before GELU:\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << "A[" << i << "]: " << A[i] << std::endl;
    }

    // Allocate device memory
    float* d_A;
    HIP_CHECK(hipMalloc(&d_A, N * sizeof(float)));
    
    // Copy data to device
    HIP_CHECK(hipMemcpy(d_A, A, N * sizeof(float), hipMemcpyHostToDevice));

    // Setup timing
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    // Record start time
    HIP_CHECK(hipEventRecord(start));

    // Launch kernel
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x);
    hipLaunchKernelGGL(gelu_kernel, blocksPerGrid, threadsPerBlock, 0, 0, d_A, N);

    // Check for kernel launch errors
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // Record stop time
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float milliseconds = 0;
    HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));
    std::cout << "\nHIP kernel time: " << milliseconds / 1000.0 << " seconds" << std::endl;

    // Copy result back to host
    HIP_CHECK(hipMemcpy(A, d_A, N * sizeof(float), hipMemcpyDeviceToHost));

    // Print first 10 elements after GELU
    std::cout << "\nAfter GELU:\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << "A[" << i << "]: " << A[i] << std::endl;
    }

    // Clean up events
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    // Free device memory
    HIP_CHECK(hipFree(d_A));

    // Free host memory
    delete[] A;

    return 0;
} 