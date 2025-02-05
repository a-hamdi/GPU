
#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define BLOCK_SIZE 256

// CUDA kernel to compute predictions and squared loss
__global__ void compute_loss(float* X, float* y, float* W, float* b, float* loss, float* y_pred, int N, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {  
        float y_pred_val = 0.0f;
        for (int i = 0; i < D; i++) {
            y_pred_val += X[idx * D + i] * W[i];
        }
        y_pred_val += *b; // Use single scalar bias
        y_pred[idx] = y_pred_val;
        loss[idx] = (y[idx] - y_pred_val) * (y[idx] - y_pred_val); // Squared loss
    }
}

// CUDA kernel to compute gradients
__global__ void compute_gradients(float* X, float* loss, float* dW, float* db, int N, int D) {
    __shared__ float db_shared[BLOCK_SIZE];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < D) {
        float gradW = 0.0f;
        for (int i = 0; i < N; i++) {
            gradW += X[i * D + idx] * loss[i];
        }
        dW[idx] = - (2.0f / N) * gradW;
    }
    float gradb = 0.0f;
    if (idx < N) {
        gradb = loss[idx];
    }
    db_shared[threadIdx.x] = gradb;
    __syncthreads();

    if (threadIdx.x == 0) {
        float sum_db = 0.0f;
        for (int i = 0; i < blockDim.x; i++) {
            sum_db += db_shared[i];
        }
        atomicAdd(db, - (2.0f / N) * sum_db);
    }
}

// CUDA kernel to update weights using SGD
__global__ void update_weights(float* W, float* dW, float* b, float* db, float lr, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < D) {
        W[idx] -= lr * dW[idx];
    }
    if (idx == 0) {
        *b -= lr * (*db);
    }
}

// Host function to train the model
void train_sgd(float* h_X, float* h_y, float* h_W, float* h_b, int N, int D, float lr, int epochs) {
    float *d_X, *d_y, *d_W, *d_b, *d_gradW, *d_gradb, *d_loss, *d_y_pred;
    cudaMalloc(&d_X, N * D * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMalloc(&d_W, D * sizeof(float));
    cudaMalloc(&d_b, sizeof(float));
    cudaMalloc(&d_gradW, D * sizeof(float));
    cudaMalloc(&d_gradb, sizeof(float));
    cudaMalloc(&d_loss, N * sizeof(float));
    cudaMalloc(&d_y_pred, N * sizeof(float));
    
    cudaMemcpy(d_X, h_X, N * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, h_W, D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float), cudaMemcpyHostToDevice);
    
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int blocks_grad = (D + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        compute_loss<<<blocks, BLOCK_SIZE>>>(d_X, d_y, d_W, d_b, d_loss, d_y_pred, N, D);
        cudaDeviceSynchronize();

        compute_gradients<<<blocks_grad, BLOCK_SIZE>>>(d_X, d_loss, d_gradW, d_gradb, N, D);
        cudaDeviceSynchronize();

        update_weights<<<blocks_grad, BLOCK_SIZE>>>(d_W, d_gradW, d_b, d_gradb, lr, D);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_W, d_W, D * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, d_b, sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_W);
    cudaFree(d_b);
    cudaFree(d_gradW);
    cudaFree(d_gradb);
    cudaFree(d_loss);
    cudaFree(d_y_pred);
}

int main() {
    int N = 1024;
    int D = 10;
    float lr = 0.01;
    int epochs = 1000;

    float *h_X = new float[N * D];
    float *h_y = new float[N];
    float *h_W = new float[D];
    float *h_b = new float[1];

    srand(42);
    for (int i = 0; i < N * D; i++) {
        h_X[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < N; i++) {
        h_y[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < D; i++) {
        h_W[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    *h_b = static_cast<float>(rand()) / RAND_MAX;

    train_sgd(h_X, h_y, h_W, h_b, N, D, lr, epochs);

    std::cout << "Trained Weights: ";
    for (int i = 0; i < D; i++) std::cout << h_W[i] << " ";
    std::cout << "\nTrained Bias: " << *h_b << std::endl;

    delete[] h_X;
    delete[] h_y;
    delete[] h_W;
    delete[] h_b;
    return 0;
}
