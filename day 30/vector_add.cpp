#include <hip/hip_runtime.h>
#include <iostream>

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int N = 10;
    float A[N], B[N], C[N];

    // Initialize input arrays
    for(int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = i * 2;
    }

    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, N * sizeof(float));
    hipMalloc(&d_b, N * sizeof(float));
    hipMalloc(&d_c, N * sizeof(float));

    hipMemcpy(d_a, A, N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_b, B, N * sizeof(float), hipMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;  // Ceiling division
    
    hipLaunchKernelGGL(vectorAdd, 
                       dim3(gridSize), 
                       dim3(blockSize),
                       0, 0,
                       d_a, d_b, d_c, N);

    hipMemcpy(C, d_c, N * sizeof(float), hipMemcpyDeviceToHost);

    // Print results
    for(int i = 0; i < N; i++) {
        std::cout << A[i] << " + " << B[i] << " = " << C[i] << std::endl;
    }

    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);

    return 0;
} 