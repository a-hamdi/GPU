#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <random>

// Kernel function for SwiGLU
__global__ void swiglu_kernel(float* out, const float* x, const float* W1, const float* W2, int batch_size, int hidden_dim, int output_dim) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int o = blockIdx.y * blockDim.y + threadIdx.y;

    if (b < batch_size && o < output_dim) {
        float xW1 = 0.0f;
        float xW2 = 0.0f;
        
        for (int i = 0; i < hidden_dim; i++) {
            xW1 += x[b * hidden_dim + i] * W1[o + i * output_dim];
            xW2 += x[b * hidden_dim + i] * W2[o + i * output_dim];
        }
        
        float sigmoid_val = 1.0f / (1.0f + expf(-xW1));
        float result = xW1 * sigmoid_val * xW2;
        
        if (b == 0 && o == 0) {  // Print debug info for first element
            printf("GPU Debug: xW1=%f, xW2=%f, sigmoid_val=%f, result=%f\n", 
                   xW1, xW2, sigmoid_val, result);
        }
        
        out[b * output_dim + o] = result;
    }
}

void swiglu_forward(float* out, const float* x, const float* W1, const float* W2, int batch_size, int hidden_dim, int output_dim) {
    // Allocate memory on GPU
    float *d_x, *d_W1, *d_W2, *d_out;
    cudaMalloc((void**)&d_x, batch_size * hidden_dim * sizeof(float));
    cudaMalloc((void**)&d_W1, hidden_dim * output_dim * sizeof(float));
    cudaMalloc((void**)&d_W2, hidden_dim * output_dim * sizeof(float));
    cudaMalloc((void**)&d_out, batch_size * output_dim * sizeof(float));
    
    // Copy data to GPU
    cudaMemcpy(d_x, x, batch_size * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, W1, hidden_dim * output_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, W2, hidden_dim * output_dim * sizeof(float), cudaMemcpyHostToDevice);
    
    // Define CUDA kernel launch parameters
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((batch_size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (output_dim + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    // Launch kernel
    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_x, d_W1, d_W2, batch_size, hidden_dim, output_dim);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }
    
    // Copy result back to CPU
    cudaMemcpy(out, d_out, batch_size * output_dim * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free GPU memory
    cudaFree(d_x);
    cudaFree(d_W1);
    cudaFree(d_W2);
    cudaFree(d_out);
}

int main() {
    int batch_size = 32;
    int hidden_dim = 128;
    int output_dim = 64;
    
    // Allocate memory
    float *x = new float[batch_size * hidden_dim];
    float *W1 = new float[hidden_dim * output_dim];
    float *W2 = new float[hidden_dim * output_dim];
    float *out = new float[batch_size * output_dim];
    
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    // Initialize input data with random values between 0 and 1
    for (int i = 0; i < batch_size * hidden_dim; i++) {
        x[i] = dis(gen);
    }
    for (int i = 0; i < hidden_dim * output_dim; i++) {
        W1[i] = dis(gen);
        W2[i] = dis(gen);
    }
    
    // Manual CPU calculation for first element (for verification)
    float manual_xW1 = 0.0f;
    float manual_xW2 = 0.0f;
    for (int i = 0; i < hidden_dim; i++) {
        manual_xW1 += x[i] * W1[i * output_dim];
        manual_xW2 += x[i] * W2[i * output_dim];
    }
    std::cout << "CPU Manual calculation for first element:" << std::endl;
    std::cout << "xW1: " << manual_xW1 << std::endl;
    std::cout << "xW2: " << manual_xW2 << std::endl;
    float manual_sigmoid = 1.0f / (1.0f + exp(-manual_xW1));
    float manual_result = manual_xW1 * manual_sigmoid * manual_xW2;
    std::cout << "Expected result: " << manual_result << std::endl;
    
    // Compute SwiGLU
    swiglu_forward(out, x, W1, W2, batch_size, hidden_dim, output_dim);
    
    // Print some input values
    std::cout << "\nFirst 10 input values:" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << "x[" << i << "]: " << x[i] << std::endl;
    }
    
    std::cout << "\nFirst 10 W1 values:" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << "W1[" << i << "]: " << W1[i] << std::endl;
    }
    
    std::cout << "\nFirst 10 W2 values:" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << "W2[" << i << "]: " << W2[i] << std::endl;
    }
    
    // Print output values
    std::cout << "\nFirst 10 output values:" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << "out[" << i << "]: " << out[i] << std::endl;
    }
    
    // Free memory
    delete[] x;
    delete[] W1;
    delete[] W2;
    delete[] out;
    
    return 0;
}
