#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

__global__ void geglu_forward_kernel(const float* __restrict__ a,
                                       const float* __restrict__ b,
                                       float* __restrict__ c,
                                       int n_cols) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (col < n_cols) {
        int idx = row * n_cols + col;
        float a_val = a[idx];
        float b_val = b[idx];
        const float sqrt_2_over_pi = 0.7978845608028654f;
        float a_cubed = a_val * a_val * a_val;
        float tanh_arg = sqrt_2_over_pi * (a_val + 0.044715f * a_cubed);
        float tanh_result = tanhf(tanh_arg);
        float geglu_a = 0.5f * a_val * (1.0f + tanh_result);
        c[idx] = geglu_a * b_val;
    }
}

__global__ void geglu_backward_kernel(const float* __restrict__ dc,
                                        const float* __restrict__ a,
                                        const float* __restrict__ b,
                                        float* __restrict__ da,
                                        float* __restrict__ db,
                                        int n_cols) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (col < n_cols) {
        int idx = row * n_cols + col;
        float dc_val = dc[idx];
        float a_val = a[idx];
        float b_val = b[idx];
        const float sqrt_2_over_pi = 0.7978845608028654f;
        float a_cubed = a_val * a_val * a_val;
        float tanh_arg = sqrt_2_over_pi * (a_val + 0.044715f * a_cubed);
        float tanh_result = tanhf(tanh_arg);
        float geglu_a = 0.5f * a_val * (1.0f + tanh_result);
        db[idx] = dc_val * geglu_a;
        float term1 = 0.5f * (1.0f + tanh_result);
        float tanh_sq = tanh_result * tanh_result;
        float term2 = 0.5f * a_val * (1.0f - tanh_sq) *
                      (sqrt_2_over_pi * (1.0f + 3.0f * 0.044715f * a_val * a_val));
        da[idx] = dc_val * b_val * (term1 + term2);
    }
}

void run_geglu_example(const float* h_a, const float* h_b, float* h_c,
                       int n_rows, int n_cols) {
    size_t num_elements = n_rows * n_cols;
    size_t bytes = num_elements * sizeof(float);
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    geglu_forward_kernel<<<n_rows, n_cols>>>(d_a, d_b, d_c, n_cols);
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main() {
    const int n_rows = 2;
    const int n_cols = 8;
    float h_a[n_rows * n_cols] = { };
    float h_b[n_rows * n_cols] = { };
    float h_c[n_rows * n_cols] = {0};
    for (int i = 0; i < n_rows * n_cols; i++) {
        h_a[i] = 0.1f * i;
        h_b[i] = 0.2f * i;
    }
    run_geglu_example(h_a, h_b, h_c, n_rows, n_cols);
    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            printf("%f ", h_c[i * n_cols + j]);
        }
        printf("\n");
    }
    return 0;
}
