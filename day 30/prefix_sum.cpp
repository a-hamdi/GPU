#include <hip/hip_runtime.h>
#include <iostream>

#define LOAD_SIZE 32

__global__ void prefixsum_kernel(float* A, float* C, int N) {
    int threadId = threadIdx.x;
    int i = 2 * blockDim.x * blockIdx.x + threadId;

    // Load in shared memory
    __shared__ float S_A[LOAD_SIZE];
    if (i < N) {
        S_A[threadId] = A[i];
    }
    if (i + blockDim.x < N) {
        S_A[threadId + blockDim.x] = A[i + blockDim.x];
    }
    __syncthreads();

    // Up-sweep (reduce) phase
    for (int jump = 1; jump <= blockDim.x; jump *= 2) {
        __syncthreads();
        int j = jump * 2 * (threadId + 1) - 1;
        if (j < LOAD_SIZE) {
            S_A[j] += S_A[j - jump];
        }
    }
    __syncthreads();

    // Down-sweep phase
    for (int jump = LOAD_SIZE/4; jump >= 1; jump /= 2) {
        __syncthreads();
        int j = jump * 2 * (threadId + 1) - 1;
        if (j < LOAD_SIZE - jump) {
            S_A[j + jump] += S_A[j];
        }
        __syncthreads();
    }

    // Store results back to global memory
    if (i < N) {
        C[i] = S_A[threadId];
    }
    if (i + blockDim.x < N) {
        C[i + blockDim.x] = S_A[threadId + blockDim.x];
    }
    __syncthreads();
}

void checkHipError(const char* message) {
    hipError_t error = hipGetLastError();
    if (error != hipSuccess) {
        printf("HIP error (%s): %s\n", message, hipGetErrorString(error));
        exit(-1);
    }
}

int main() {
    const int N = 10;
    float A[N], C[N];

    // Initialize input array
    for (int i = 0; i < N; i++) {
        A[i] = i + 1.0f;
    }

    // Print input array
    printf("Input Array A:\n");
    for (int i = 0; i < N; i++) {
        printf("%.2f ", A[i]);
    }
    printf("\n");

    // Allocate device memory
    float *d_A, *d_C;
    hipMalloc(&d_A, N * sizeof(float));
    hipMalloc(&d_C, N * sizeof(float));

    // Copy input data to device
    hipMemcpy(d_A, A, N * sizeof(float), hipMemcpyHostToDevice);
    checkHipError("Failed to copy input data to device");

    // Launch kernel
    dim3 dimBlock(16);  // Using 16 threads per block for this small example
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x);

    hipLaunchKernelGGL(prefixsum_kernel,
                       dimGrid,
                       dimBlock,
                       0, 0,
                       d_A, d_C, N);
    
    checkHipError("Failed to execute the kernel");
    hipDeviceSynchronize();

    // Copy results back to host
    hipMemcpy(C, d_C, N * sizeof(float), hipMemcpyDeviceToHost);
    checkHipError("Failed to copy output data to host");

    // Print results
    printf("\nPrefix Sum Results C:\n");
    for (int i = 0; i < N; i++) {
        printf("%.2f ", C[i]);
    }
    printf("\n");

    // Verify results
    printf("\nVerification:\n");
    float sum = 0.0f;
    bool correct = true;
    for (int i = 0; i < N; i++) {
        sum += A[i];
        if (fabs(C[i] - sum) > 1e-5) {
            printf("Mismatch at position %d: expected %.2f, got %.2f\n", i, sum, C[i]);
            correct = false;
        }
    }
    if (correct) {
        printf("All results match the expected values!\n");
    }

    // Cleanup
    hipFree(d_A);
    hipFree(d_C);

    return 0;
} 