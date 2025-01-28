#include "cuda_kernels.h"
#include "helper_functions.h"

__global__ void addBiasKernel(float* output, const float* bias, int batch_size, int output_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = idx / output_features;
    int feature = idx % output_features;

    if (batch < batch_size && feature < output_features) {
        output[batch * output_features + feature] += bias[feature];
    }
}

void performLinearLayerOperation(
    cublasHandle_t cublas_handle,
    const float* input_data,
    const float* weights_data,
    const float* bias_data,
    float* output_data,
    int batch_size,
    int input_features,
    int output_features
) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    checkCublasStatus(cublasSgemm(
        cublas_handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        output_features,
        batch_size,
        input_features,
        &alpha,
        weights_data,
        input_features,
        input_data,
        input_features,
        &beta,
        output_data,
        output_features
    ));

    int total_elements = batch_size * output_features;
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    addBiasKernel<<<num_blocks, block_size>>>(output_data, bias_data, batch_size, output_features);
    checkCudaStatus(cudaGetLastError());
    checkCudaStatus(cudaDeviceSynchronize());
}
