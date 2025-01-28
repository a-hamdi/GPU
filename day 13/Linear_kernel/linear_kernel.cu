#include "helper_functions.h"
#include "cuda_kernels.h"

int main() {
    // Layer dimensions
    const int batch_size = 1;
    const int input_features = 256;
    const int output_features = 256;

    // Memory sizes
    const size_t input_size = batch_size * input_features;
    const size_t weights_size = input_features * output_features;
    const size_t bias_size = output_features;
    const size_t output_size = batch_size * output_features;

    // Host memory allocation
    float* host_input = new float[input_size];
    float* host_weights = new float[weights_size];
    float* host_bias = new float[bias_size];
    float* host_output = new float[output_size];

    // Initialize host data
    initializeRandomMatrix(host_input, input_size, -1.0f, 1.0f);
    initializeRandomMatrix(host_weights, weights_size, -1.0f, 1.0f);
    initializeRandomMatrix(host_bias, bias_size, -0.1f, 0.1f);

    // Device memory allocation
    float *device_input, *device_weights, *device_bias, *device_output;
    allocateDeviceMemory(&device_input, input_size);
    allocateDeviceMemory(&device_weights, weights_size);
    allocateDeviceMemory(&device_bias, bias_size);
    allocateDeviceMemory(&device_output, output_size);

    // Copy data to device
    cudaMemcpy(device_input, host_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_weights, host_weights, weights_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_bias, host_bias, bias_size * sizeof(float), cudaMemcpyHostToDevice);

    // cuBLAS handle creation
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    // Perform linear operation
    performLinearLayerOperation(cublas_handle, device_input, device_weights, device_bias, device_output, batch_size, input_features, output_features);

    // Copy result back to host
    cudaMemcpy(host_output, device_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    cublasDestroy(cublas_handle);
    freeDeviceMemory(device_input);
    freeDeviceMemory(device_weights);
    freeDeviceMemory(device_bias);
    freeDeviceMemory(device_output);
    delete[] host_input;
    delete[] host_weights;
    delete[] host_bias;
    delete[] host_output;

    return 0;
}
