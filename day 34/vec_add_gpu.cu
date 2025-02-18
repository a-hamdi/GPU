#include <stdio.h>
#include <cuda.h>

#define UNROLL_FACTOR 4

__global__ void vector_add(double *A, double *B, double *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int limit = N - (N % UNROLL_FACTOR);

    for (; i < limit; i += blockDim.x * gridDim.x) {
        C[i] = A[i] + B[i];
        C[i+1] = A[i+1] + B[i+1];
        C[i+2] = A[i+2] + B[i+2];
        C[i+3] = A[i+3] + B[i+3];
    }

    for (; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char **argv) {
    int N = atoi(argv[1]);
    double *h_A, *h_B, *h_C;
    double *d_A, *d_B, *d_C;

    size_t size = N * sizeof(double);
    h_A = (double*) malloc(size);
    h_B = (double*) malloc(size);
    h_C = (double*) malloc(size);

    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("GPU (CUDA): %f ms\n", elapsedTime);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
