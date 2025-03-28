#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

#define BLOCK_SIZE 256

__device__ float tanh_activation(float x) {
    return tanhf(x);
}

__global__ void dyt_fwd_kernel(
    const float* x, float* y,
    const float alpha,
    const float* gamma, const float* beta,
    int n_cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_cols) {
        float val = alpha * x[idx];
        y[idx] = gamma[idx] * tanh_activation(val) + beta[idx];
    }
}

__global__ void dyt_bwd_kernel(
    const float* x, const float* dy,
    float* dx, float* dalpha, float* dgamma, float* dbeta,
    float alpha, const float* gamma,
    int n_cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_cols) {
        float tanh_ax = tanh_activation(alpha * x[idx]);
        float sech2_ax = 1 - tanh_ax * tanh_ax;
        
        dx[idx] = dy[idx] * gamma[idx] * sech2_ax * alpha;
        atomicAdd(dalpha, dy[idx] * gamma[idx] * sech2_ax * x[idx]);
        atomicAdd(&dgamma[idx], dy[idx] * tanh_ax);
        atomicAdd(&dbeta[idx], dy[idx]);
    }
}

void liger_dyt_fwd(float* x, float* y, float alpha, float* gamma, float* beta, int n_cols) {
    float *d_x, *d_y, *d_gamma, *d_beta;
    cudaMalloc(&d_x, n_cols * sizeof(float));
    cudaMalloc(&d_y, n_cols * sizeof(float));
    cudaMalloc(&d_gamma, n_cols * sizeof(float));
    cudaMalloc(&d_beta, n_cols * sizeof(float));

    cudaMemcpy(d_x, x, n_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma, n_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, beta, n_cols * sizeof(float), cudaMemcpyHostToDevice);
    
    dyt_fwd_kernel<<<(n_cols + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_x, d_y, alpha, d_gamma, d_beta, n_cols);
    cudaMemcpy(y, d_y, n_cols * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_gamma);
    cudaFree(d_beta);
}

void liger_dyt_bwd(float* x, float* dy, float* dx, float* dalpha, float* dgamma, float* dbeta, float alpha, float* gamma, int n_cols) {
    float *d_x, *d_dy, *d_dx, *d_dalpha, *d_dgamma, *d_dbeta, h_dalpha = 0;
    cudaMalloc(&d_x, n_cols * sizeof(float));
    cudaMalloc(&d_dy, n_cols * sizeof(float));
    cudaMalloc(&d_dx, n_cols * sizeof(float));
    cudaMalloc(&d_dalpha, sizeof(float));
    cudaMalloc(&d_dgamma, n_cols * sizeof(float));
    cudaMalloc(&d_dbeta, n_cols * sizeof(float));

    cudaMemcpy(d_x, x, n_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dy, dy, n_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dalpha, &h_dalpha, sizeof(float), cudaMemcpyHostToDevice);
    
    dyt_bwd_kernel<<<(n_cols + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_x, d_dy, d_dx, d_dalpha, d_dgamma, d_dbeta, alpha, gamma, n_cols);
    
    cudaMemcpy(dx, d_dx, n_cols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dalpha, d_dalpha, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dgamma, d_dgamma, n_cols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dbeta, d_dbeta, n_cols * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_x);
    cudaFree(d_dy);
    cudaFree(d_dx);
    cudaFree(d_dalpha);
    cudaFree(d_dgamma);
    cudaFree(d_dbeta);
}

int main() {
    const int n_cols = 8;
    float x[n_cols] = {0.5, -0.3, 0.7, 1.2, -1.5, 0.9, -0.4, 0.2};
    float gamma[n_cols] = {1, 1, 1, 1, 1, 1, 1, 1};
    float beta[n_cols] = {0, 0, 0, 0, 0, 0, 0, 0};
    float alpha = 0.8;
    float y[n_cols], dx[n_cols], dalpha, dgamma[n_cols], dbeta[n_cols];

    liger_dyt_fwd(x, y, alpha, gamma, beta, n_cols);
    
    std::cout << "Forward Output: ";
    for (int i = 0; i < n_cols; ++i) {
        std::cout << y[i] << " ";
    }
    std::cout << std::endl;

    float dy[n_cols] = {1, 1, 1, 1, 1, 1, 1, 1};
    liger_dyt_bwd(x, dy, dx, &dalpha, dgamma, dbeta, alpha, gamma, n_cols);
    
    std::cout << "Backward dx: ";
    for (int i = 0; i < n_cols; ++i) {
        std::cout << dx[i] << " ";
    }
    std::cout << "\nBackward dalpha: " << dalpha << std::endl;
    return 0;
}
