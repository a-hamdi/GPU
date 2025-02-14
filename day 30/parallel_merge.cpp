#include <hip/hip_runtime.h>
#include <iostream>

#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error in %s:%d: %s\n", __FILE__, __LINE__, \
                hipGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__device__ void co_rank(const int* A, const int* B, int k, const int N, const int M, int* i_out, int* j_out) {
    int low = max(0, k-M);
    int high = min(k, N);
    
    while (low <= high) {
        int i = (low + high) / 2;
        int j = k - i;
        
        if (j < 0) {
            high = i - 1;
            continue;
        }
        if (j > M) {
            low = i + 1;
            continue;
        }

        if (i > 0 && j < M && A[i-1] > B[j]) {
            high = i - 1;
        }
        else if (j > 0 && i < N && B[j-1] > A[i]) {
            low = i + 1;
        }
        else {
            *i_out = i;
            *j_out = j;
            return;
        }
    }
}

__global__ void parallel_merge(const int* A, const int* B, int* C, const int N, const int M) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < N + M) {
        int i, j;
        co_rank(A, B, tid, N, M, &i, &j);
        
        if (j >= M || (i < N && A[i] <= B[j])) {
            C[tid] = A[i];
        } else {
            C[tid] = B[j];
        }
    }
}

int main() {
    const int N = 5;
    const int M = 5;
    int A[N], B[M], C[N+M];
    
    // Initialize arrays with sorted values
    for(int i = 0; i < N; i++) {
        A[i] = 2*i;  // Even numbers: 0,2,4,6,8
    }
    for(int i = 0; i < M; i++) {
        B[i] = 2*i + 1;  // Odd numbers: 1,3,5,7,9
    }

    std::cout << "Array A: ";
    for(int i = 0; i < N; i++) {
        std::cout << A[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Array B: ";
    for(int i = 0; i < M; i++) {
        std::cout << B[i] << " ";
    }
    std::cout << std::endl;

    // Declare device pointers
    int *d_A, *d_B, *d_C;
    
    // Allocate memory on device
    HIP_CHECK(hipMalloc(&d_A, N * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_B, M * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_C, (N+M) * sizeof(int)));
    
    // Copy data from host to device
    HIP_CHECK(hipMemcpy(d_A, A, N * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, B, M * sizeof(int), hipMemcpyHostToDevice));

    // Setup timing
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    // Record start time
    HIP_CHECK(hipEventRecord(start));

    // Set up execution configuration
    dim3 block(256);
    dim3 grid((N+M + block.x-1) / block.x);
    
    // Launch kernel
    hipLaunchKernelGGL(parallel_merge, grid, block, 0, 0, d_A, d_B, d_C, N, M);
    
    // Check for kernel launch errors
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // Record stop time
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float milliseconds = 0;
    HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));
    std::cout << "HIP kernel time: " << milliseconds / 1000.0 << " seconds" << std::endl;
    
    // Copy result back to host
    HIP_CHECK(hipMemcpy(C, d_C, (N+M) * sizeof(int), hipMemcpyDeviceToHost));
    
    // Print result
    std::cout << "Merged array: ";
    for(int i = 0; i < N+M; i++) {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;

    // Clean up events
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    
    // Free device memory
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));

    return 0;
} 