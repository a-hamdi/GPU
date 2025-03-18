#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>


__global__ void normalize_vector(float* vec, int size) {
    float norm = 0.0f;
    for (int i = 0; i < size; ++i) {
        norm += vec[i] * vec[i];
    }
    norm = sqrtf(norm);
    for (int i = 0; i < size; ++i) {
        vec[i] /= norm;
    }
}


void matvec_mul(cublasHandle_t handle, const float* matrix, const float* vec, float* result, int rows, int cols) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemv(handle, CUBLAS_OP_N, rows, cols, &alpha, matrix, rows, vec, 1, &beta, result, 1);
}

float power_iteration(const float* d_matrix, int rows, int cols, int num_iterations) {

    float* d_u;
    float* d_v;
    cudaMalloc(&d_u, rows * sizeof(float));
    cudaMalloc(&d_v, cols * sizeof(float));

    std::vector<float> h_v(cols, 1.0f); 
    cudaMemcpy(d_v, h_v.data(), cols * sizeof(float), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    for (int i = 0; i < num_iterations; ++i) {
       
        matvec_mul(handle, d_matrix, d_v, d_u, rows, cols);

   
        normalize_vector<<<1, 1>>>(d_u, rows);
        cudaDeviceSynchronize();


        matvec_mul(handle, d_matrix, d_u, d_v, cols, rows);

        normalize_vector<<<1, 1>>>(d_v, cols);
        cudaDeviceSynchronize();
    }

    float* d_Wv;
    cudaMalloc(&d_Wv, rows * sizeof(float));
    matvec_mul(handle, d_matrix, d_v, d_Wv, rows, cols);

    float sigma;
    cublasSnrm2(handle, rows, d_Wv, 1, &sigma);


    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_Wv);
    cublasDestroy(handle);

    return sigma;
}

int main() {

    const int rows = 4;
    const int cols = 3;
    const int num_iterations = 10;

    std::vector<float> h_matrix = {
        0.5f, 0.2f, 0.1f,
        0.4f, 0.3f, 0.6f,
        0.7f, 0.5f, 0.8f,
        0.9f, 0.4f, 0.2f
    };

 
    float* d_matrix;
    cudaMalloc(&d_matrix, rows * cols * sizeof(float));
    cudaMemcpy(d_matrix, h_matrix.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);


    float sigma = power_iteration(d_matrix, rows, cols, num_iterations);
    std::cout << "Spectral Norm: " << sigma << std::endl;


    for (auto& val : h_matrix) {
        val /= sigma;
    }


    cudaFree(d_matrix);

    return 0;
}
