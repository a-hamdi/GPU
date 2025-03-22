#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

__device__ inline float sigmoid(float x) {
    return 1.f / (1.f + expf(-x));
}

__device__ inline float silu(float x) {
    return x * sigmoid(x);
}

__global__ void swiglu_forward_kernel(const float *a, const float *b, float *c, int stride, int n_cols) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    
    if (col < n_cols) {
        int offset = row * stride + col;
        float a_val = a[offset];
        float b_val = b[offset];
        c[offset] = silu(a_val) * b_val;
    }
}

__global__ void swiglu_backward_kernel(float *dc, float *a, float *b, int stride, int n_cols) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    
    if (col < n_cols) {
        int offset = row * stride + col;
        float dc_val = dc[offset];
        float a_val = a[offset];
        float b_val = b[offset];
        
        float sig_a = sigmoid(a_val);
        float silu_a = a_val * sig_a;
        float db = dc_val * silu_a;
        float da = dc_val * (silu_a * (1.f - sig_a) + sig_a) * b_val;
        
        a[offset] = da;  
        b[offset] = db;
    }
}

int main() {
    const int n_rows = 4;
    const int n_cols = 8;
    const int stride = n_cols;
    const size_t num_elements = n_rows * stride;
    const size_t size = num_elements * sizeof(float);
    
    float h_a[num_elements];
    float h_b[num_elements];
    float h_c[num_elements];
    float h_dc[num_elements];
    
    for (int i = 0; i < num_elements; i++) {
        h_a[i] = static_cast<float>(i) / 10.0f;
        h_b[i] = 1.0f;
        h_dc[i] = 1.0f;
    }
    
    float *d_a, *d_b, *d_c, *d_dc;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    cudaMalloc(&d_dc, size);
    
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dc, h_dc, size, cudaMemcpyHostToDevice);
    
    swiglu_forward_kernel<<<n_rows, n_cols>>>(d_a, d_b, d_c, stride, n_cols);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    std::cout << "Forward pass result (c = silu(a)*b):" << std::endl;
    for (int row = 0; row < n_rows; row++) {
        for (int col = 0; col < n_cols; col++) {
            std::cout << h_c[row * stride + col] << " ";
        }
        std::cout << std::endl;
    }
    
    swiglu_backward_kernel<<<n_rows, n_cols>>>(d_dc, d_a, d_b, stride, n_cols);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost);
    
    std::cout << "\nBackward pass gradients:" << std::endl;
    
    std::cout << "Gradient for a:" << std::endl;
    for (int row = 0; row < n_rows; row++) {
        for (int col = 0; col < n_cols; col++) {
            std::cout << h_a[row * stride + col] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\nGradient for b:" << std::endl;
    for (int row = 0; row < n_rows; row++) {
        for (int col = 0; col < n_cols; col++) {
            std::cout << h_b[row * stride + col] << " ";
        }
        std::cout << std::endl;
    }
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_dc);
    
    return 0;
}
