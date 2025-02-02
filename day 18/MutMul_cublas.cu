#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);

    int M = 2, N = 3, K = 4;
    float *h_A, *h_B, *h_C;
    h_A = (float *)malloc(M * K * sizeof(float));
    h_B = (float *)malloc(K * N * sizeof(float));
    h_C = (float *)malloc(M * N * sizeof(float));

    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            h_A[i * K + j] = i + j;

    for (int i = 0; i < K; i++)
        for (int j = 0; j < N; j++)
            h_B[i * N + j] = i + j;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                M, N, K, &alpha,
                d_A, M, d_B, K,
                &beta, d_C, M);

    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Matrix A:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            printf("%f ", h_A[i * K + j]);
        }
        printf("\n");
    }

    printf("Matrix B:\n");
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", h_B[i * N + j]);
        }
        printf("\n");
    }

    printf("Matrix C = A * B:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", h_C[i + j * M]);
        }
        printf("\n");
    }

    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cublasDestroy(handle);
    return 0;
}
