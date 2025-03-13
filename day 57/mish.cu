#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

// CUDA Kernel for Mish Activation Function
__global__ void mish_kernel(float* x, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        float softplus = logf(1 + expf(val));  // softplus(x)
        y[idx] = val * tanhf(softplus);        // mish(x)
    }
}

// Host function to launch the kernel
void mish_cuda(float* d_x, float* d_y, int size) {
    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;
    mish_kernel<<<blocks_per_grid, threads_per_block>>>(d_x, d_y, size);
    cudaDeviceSynchronize();
}

// Main Function
int main() {
    const int size = 10;  // Number of elements
    size_t bytes = size * sizeof(float);

    // Allocate host memory
    float* h_x = new float[size];
    float* h_y = new float[size];

    // Initialize input data
    for (int i = 0; i < size; i++) {
        h_x[i] = (float)(i - 5);  // Values from -5 to 4
    }

    // Allocate device memory
    float *d_x, *d_y;
    cudaMalloc(&d_x, size * sizeof(float));
    cudaMalloc(&d_y, size * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_x, h_x, size * sizeof(float), cudaMemcpyHostToDevice);

    // Define CUDA kernel launch parameters
    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

    // Launch the kernel
    mish_cuda<<<blocks_per_grid, threads_per_block>>>(d_x, d_y, size);

    // Copy result from device to host
    cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost);

    // Print the results
    std::cout << "Input\tMish Output\n";
    for (int i = 0; i < size; i++) {
        std::cout << "Mish(" << h_x[i] << ") = " << h_y[i] << std::endl;
    }

    // Free GPU memory
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
