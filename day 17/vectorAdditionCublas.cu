// nvcc vec_cublas.cu -o vec_cublas -lstdc++ -lcublas

#include <iostream>
#include <cublas_v2.h>

int main() {
    const int N = 10;
    float A[N], B[N], C[N];

    // Initialize input vectors (you might want to add your own initialization)
    for(int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = i;
    }

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Allocate device memory
    float *d_a, *d_b;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_a, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Scaling factors
    const float alpha = 1.0f;

    // Perform vector addition: C = alpha*A + B
    cublasSaxpy(handle, N, &alpha, d_a, 1, d_b, 1);

    // Copy result back to host (result is in d_b)
    cudaMemcpy(C, d_b, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results
    for(int i = 0; i < N; i++) {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cublasDestroy(handle);

    return 0;
}
