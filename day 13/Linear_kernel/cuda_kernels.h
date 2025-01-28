#pragma once

__global__ void addBiasKernel(float* output, const float* bias, int batch_size, int output_features);
void performLinearLayerOperation(cublasHandle_t cublas_handle, const float* input_data, const float* weights_data, const float* bias_data, float* output_data, int batch_size, int input_features, int output_features);
