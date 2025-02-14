#include <hip/hip_runtime.h>
#include <iostream>
#include <cmath>

__global__ void MatrixAdd_C(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        for(int j = 0; j < N; j++) {
            C[i*N+j] = A[i*N+j] + B[i*N+j];
        }
    }
}

__global__ void MatrixAdd_B(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i >= N) || (j >= N)) { return; }  // Fixed the condition

    C[i*N+j] = A[i*N+j] + B[i*N+j];
}

__global__ void MatrixAdd_D(const float* A, const float* B, float* C, int N) {
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < N) {
        for(int i = 0; i < N; i++) {
            C[i*N+j] = A[i*N+j] + B[i*N+j];
        }
    }
}

int main() {
    const int N = 10;
    float *A, *B, *C;

    // Initialize the input matrices
    A = (float *)malloc(N * N * sizeof(float));
    B = (float *)malloc(N * N * sizeof(float));
    C = (float *)malloc(N * N * sizeof(float));

    // Initialize matrices with values
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = 1.0f;
            B[i * N + j] = 2.0f;
            C[i * N + j] = 0.0f;
        }
    }

    float *d_a, *d_b, *d_c;
    hipMalloc((void **)&d_a, N * N * sizeof(float));
    hipMalloc((void **)&d_b, N * N * sizeof(float));
    hipMalloc((void **)&d_c, N * N * sizeof(float));

    hipMemcpy(d_a, A, N * N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_b, B, N * N * sizeof(float), hipMemcpyHostToDevice);

    dim3 dimBlock(32, 16);
    dim3 dimGrid(ceil(N / 32.0f), ceil(N / 16.0f));
    
    // Launch kernel using hipLaunchKernelGGL
    hipLaunchKernelGGL(MatrixAdd_B,
                       dimGrid,
                       dimBlock,
                       0, 0,
                       d_a, d_b, d_c, N);
    
    hipDeviceSynchronize();

    hipMemcpy(C, d_c, N * N * sizeof(float), hipMemcpyDeviceToHost);

    // Print results
    std::cout << "C:\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f ", C[i * N + j]);
        }
        printf("\n");
    }

    std::cout << "\nA:\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f ", A[i * N + j]);
        }
        printf("\n");
    }

    std::cout << "\nB:\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f ", B[i * N + j]);
        }
        printf("\n");
    }

    // Cleanup
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
    free(A);
    free(B);
    free(C);

    return 0;
} 