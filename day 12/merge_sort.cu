#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

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

    printf("Array A: ");
    for(int i = 0; i < N; i++) {
        printf("%d ", A[i]);
    }
    printf("\n");

    printf("Array B: ");
    for(int i = 0; i < M; i++) {
        printf("%d ", B[i]);
    }
    printf("\n");

    // Declare device pointers
    int *d_A, *d_B, *d_C;
    
    // Allocate memory on device
    cudaMalloc(&d_A, N * sizeof(int));
    cudaMalloc(&d_B, M * sizeof(int));
    cudaMalloc(&d_C, (N+M) * sizeof(int));
    
    // Copy data from host to device
    cudaMemcpy(d_A, A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, M * sizeof(int), cudaMemcpyHostToDevice);

    // Set up execution configuration
    dim3 block(256);
    dim3 grid((N+M + block.x-1) / block.x);
    
    // Launch kernel
    parallel_merge<<<grid, block>>>(d_A, d_B, d_C, N, M);
    
    // Copy result back to host
    cudaMemcpy(C, d_C, (N+M) * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    // Print result
    printf("Merged array: ");
    for(int i = 0; i < N+M; i++) {
        printf("%d ", C[i]);
    }
    printf("\n");

    return 0;
}
