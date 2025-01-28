#include "helper_functions.h"

void printMatrix(const char* name, const float* matrix, int rows, int cols) {
    std::cout << name << " (" << rows << "x" << cols << "):\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(4) << matrix[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
}

void initializeRandomMatrix(float* matrix, int size, float min_val, float max_val) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    for (int i = 0; i < size; ++i) {
        matrix[i] = dis(gen);
    }
}

cudaError_t allocateDeviceMemory(float** device_pointer, size_t size) {
    return cudaMalloc(device_pointer, size);
}

void freeDeviceMemory(float* device_pointer) {
    if (device_pointer) {
        cudaFree(device_pointer);
    }
}

void checkCudaStatus(cudaError_t status) {
    if (status != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(status) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkCublasStatus(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS error: " << status << std::endl;
        exit(EXIT_FAILURE);
    }
}
