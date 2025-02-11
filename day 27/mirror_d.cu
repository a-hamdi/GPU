#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024  // Number of elements
#define ETA 0.5f // Larger learning rate

// Mirror Maps
#define EUCLIDEAN         0  // Standard gradient descent
#define NEGATIVE_ENTROPY  1  // Exponentiated gradient descent
#define LOG_BARRIER       2  // Positive orthant

__global__ void mirror_descent(float *x, float *grad, float eta, int mirror_map, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float new_x = x[i];

    switch (mirror_map) {
        case EUCLIDEAN:
            new_x = x[i] - eta * grad[i];
            break;

        case NEGATIVE_ENTROPY:
            new_x = x[i] * expf(-eta * grad[i]); // Ensure updates are visible
            break;

        case LOG_BARRIER:
            new_x = x[i] / (1.0f + eta * grad[i]);
            break;

        default:
            new_x = x[i]; 
    }

    x[i] = new_x;
}

void checkCuda(cudaError_t result, const char *msg) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s (%s)\n", msg, cudaGetErrorString(result));
        exit(-1);
    }
}

int main() {
    float *x, *grad, *d_x, *d_grad;
    int mirror_map = NEGATIVE_ENTROPY; // Choose the method

    // Allocate memory
    x = (float*)malloc(N * sizeof(float));
    grad = (float*)malloc(N * sizeof(float));
    checkCuda(cudaMalloc(&d_x, N * sizeof(float)), "Alloc d_x");
    checkCuda(cudaMalloc(&d_grad, N * sizeof(float)), "Alloc d_grad");

    // Initialize x and grad
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;  // Start with x_t = 1
        grad[i] = 0.5f * i; // Larger gradient values for visible updates
    }

    // Copy to GPU
    checkCuda(cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice), "Memcpy x -> d_x");
    checkCuda(cudaMemcpy(d_grad, grad, N * sizeof(float), cudaMemcpyHostToDevice), "Memcpy grad -> d_grad");

    // Kernel execution
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    mirror_descent<<<numBlocks, blockSize>>>(d_x, d_grad, ETA, mirror_map, N);
    checkCuda(cudaDeviceSynchronize(), "Kernel execution");

    // Copy results back
    checkCuda(cudaMemcpy(x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost), "Memcpy d_x -> x");

    // Print first 10 results
    for (int i = 0; i < 10; i++) {
        printf("x[%d] = %f\n", i, x[i]);
    }

    // Cleanup
    free(x);
    free(grad);
    cudaFree(d_x);
    cudaFree(d_grad);

    return 0;
}
