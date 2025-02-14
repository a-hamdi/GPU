#include <hip/hip_runtime.h>
#include <iostream>

__global__ void vectorMatrixMult(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += A[i*N+j] * B[j];
        }
        C[i] = sum;
    }
}

int main() {
    const int N = 10;
    float *A, *B, *C;

    // Initialize the input arrays
    A = (float *)malloc(N * N * sizeof(float));
    B = (float *)malloc(N * sizeof(float));
    C = (float *)malloc(N * sizeof(float));

    // Initialize matrices with values
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = 1.0f;
        }
        B[i] = 2.0f;
        C[i] = 0.0f;
    }

    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, N * N * sizeof(float));
    hipMalloc(&d_b, N * sizeof(float));
    hipMalloc(&d_c, N * sizeof(float));

    hipMemcpy(d_a, A, N * N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_b, B, N * sizeof(float), hipMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;  // Ceiling division
    
    // Launch kernel using hipLaunchKernelGGL
    hipLaunchKernelGGL(vectorMatrixMult,
                       dim3(gridSize),
                       dim3(blockSize),
                       0, 0,
                       d_a, d_b, d_c, N);
    
    hipDeviceSynchronize();

    hipMemcpy(C, d_c, N * sizeof(float), hipMemcpyDeviceToHost);

    // Print matrix A
    std::cout << "A:\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f ", A[i * N + j]);
        }
        printf("\n");
    }

    // Print vector C (result)
    std::cout << "\nC:\n";
    for (int i = 0; i < N; i++) {
        printf("%.2f ", C[i]);
    }
    printf("\n");

    // Print vector B
    std::cout << "\nB:\n";
    for (int i = 0; i < N; i++) {
        printf("%.2f ", B[i]);
    }
    printf("\n");

    // Cleanup
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
    free(A);
    free(B);
    free(C);

    return 0;
} 