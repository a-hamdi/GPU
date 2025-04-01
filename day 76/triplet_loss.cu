#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

__global__ void triplet_loss_kernel(const float* anchor, const float* positive, const float* negative, float* loss, float alpha, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;
    
    float ap_dist = (anchor[idx] - positive[idx]) * (anchor[idx] - positive[idx]);
    float an_dist = (anchor[idx] - negative[idx]) * (anchor[idx] - negative[idx]);
    
    float triplet_loss = fmaxf(0.0f, ap_dist - an_dist + alpha);
    atomicAdd(loss, triplet_loss);
}

void triplet_loss_cuda(const float* h_anchor, const float* h_positive, const float* h_negative, float* h_loss, float alpha, int dim) {
    float *d_anchor, *d_positive, *d_negative, *d_loss;
    cudaMalloc(&d_anchor, dim * sizeof(float));
    cudaMalloc(&d_positive, dim * sizeof(float));
    cudaMalloc(&d_negative, dim * sizeof(float));
    cudaMalloc(&d_loss, sizeof(float));
    
    cudaMemcpy(d_anchor, h_anchor, dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_positive, h_positive, dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_negative, h_negative, dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_loss, 0, sizeof(float));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (dim + threadsPerBlock - 1) / threadsPerBlock;
    triplet_loss_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_anchor, d_positive, d_negative, d_loss, alpha, dim);
    
    cudaMemcpy(h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_anchor);
    cudaFree(d_positive);
    cudaFree(d_negative);
    cudaFree(d_loss);
}

int main() {
    const int dim = 128;
    float h_anchor[dim], h_positive[dim], h_negative[dim], h_loss;
    float alpha = 0.2f;
    
    for (int i = 0; i < dim; i++) {
        h_anchor[i] = static_cast<float>(rand()) / RAND_MAX;
        h_positive[i] = static_cast<float>(rand()) / RAND_MAX;
        h_negative[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    triplet_loss_cuda(h_anchor, h_positive, h_negative, &h_loss, alpha, dim);
    
    std::cout << "Triplet Loss: " << h_loss << std::endl;
    return 0;
}
