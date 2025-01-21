#include <cuda_runtime.h>
#include <iostream>

// Define the size of the matrix
#define WIDTH 1024
#define HEIGHT 1024

// CUDA kernel for matrix transposition
__global__ void transposeMatrix(const float* input, float* output, int width, int height) {
    // Calculate the row and column index of the element
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Perform the transposition if within bounds
    if (x < width && y < height) {
        int inputIndex = y * width + x;
        int outputIndex = x * height + y;
        output[outputIndex] = input[inputIndex];
    }
}

// Host function to check for CUDA errors
void checkCudaError(const char* message) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << message << " - CUDA Error: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    int width = WIDTH;
    int height = HEIGHT;

    // Allocate host memory
    size_t size = width * height * sizeof(float);
    float* h_input = (float*)malloc(size);
    float* h_output = (float*)malloc(size);

    // Initialize the input matrix with some values
    for (int i = 0; i < width * height; i++) {
        h_input[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float* d_input;
    float* d_output;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    // Copy data from host to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    checkCudaError("Failed to copy input data to device");

    // Define block and grid sizes
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    transposeMatrix<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();
    checkCudaError("Kernel execution failed");

    // Copy the result back to the host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    checkCudaError("Failed to copy output data to host");

    // Verify the result
    bool success = true;
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            if (h_output[i * height + j] != h_input[j * width + i]) {
                success = false;
                break;
            }
        }
    }

    std::cout << (success ? "Matrix transposition succeeded!" : "Matrix transposition failed!") << std::endl;

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Free host memory
    free(h_input);
    free(h_output);

    return 0;
}