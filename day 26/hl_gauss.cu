#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>


#define DEFAULT_SIGMA_TO_BIN_RATIO 2.0f
#define PI 3.14159265358979323846f
#define EPS 1e-10f


__device__ float erf_device(float x) {
    
    float t = 1.0f / (1.0f + 0.5f * fabsf(x));
    float tau = t * expf(-x * x - 1.26551223f + t * (1.00002368f + t * (0.37409196f + 
        t * (0.09678418f + t * (-0.18628806f + t * (0.27886807f + t * (-1.13520398f + 
        t * (1.48851587f + t * (-0.82215223f + t * 0.17087277f)))))))));
    return (x >= 0) ? 1.0f - tau : tau - 1.0f;
}


__global__ void compute_histogram_probs(
    float* values,
    float* support,
    float* probs,
    int n_values,
    int n_bins,
    float sigma,
    float sigma_times_sqrt_two
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_values) return;

    float value = values[idx];
    
    
    for (int i = 0; i < n_bins; i++) {
        float cdf_right = erf_device((support[i + 1] - value) / sigma_times_sqrt_two);
        float cdf_left = erf_device((support[i] - value) / sigma_times_sqrt_two);
        probs[idx * n_bins + i] = (cdf_right - cdf_left) / 2.0f;
    }
}


extern "C" void hlgauss_loss(
    float* d_values,
    float* d_support,
    float* d_probs,
    int n_values,
    int n_bins,
    float sigma
) {
    float sigma_times_sqrt_two = sigma * sqrtf(2.0f);

    
    int block_size = 256;
    int num_blocks = (n_values + block_size - 1) / block_size;
    
    compute_histogram_probs<<<num_blocks, block_size>>>(
        d_values,
        d_support,
        d_probs,
        n_values,
        n_bins,
        sigma,
        sigma_times_sqrt_two
    );
}

int main() {
   
    const int n_values = 1000;
    const int n_bins = 10;
    const float min_value = -5.0f;
    const float max_value = 5.0f;
    const float sigma = 1.0f;

    
    float* h_values = (float*)malloc(n_values * sizeof(float));
    float* h_support = (float*)malloc((n_bins + 1) * sizeof(float));
    float* h_probs = (float*)malloc(n_values * n_bins * sizeof(float));

    
    for (int i = 0; i < n_values; i++) {
        h_values[i] = min_value + (max_value - min_value) * ((float)rand() / RAND_MAX);
    }
    
    
    float bin_width = (max_value - min_value) / n_bins;
    for (int i = 0; i <= n_bins; i++) {
        h_support[i] = min_value + i * bin_width;
    }

    
    float *d_values, *d_support, *d_probs;
    cudaMalloc(&d_values, n_values * sizeof(float));
    cudaMalloc(&d_support, (n_bins + 1) * sizeof(float));
    cudaMalloc(&d_probs, n_values * n_bins * sizeof(float));

    
    cudaMemcpy(d_values, h_values, n_values * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_support, h_support, (n_bins + 1) * sizeof(float), cudaMemcpyHostToDevice);

   
    hlgauss_loss(d_values, d_support, d_probs, n_values, n_bins, sigma);

    
    cudaMemcpy(h_probs, d_probs, n_values * n_bins * sizeof(float), cudaMemcpyDeviceToHost);

    
    printf("First few probability distributions:\n");
    for (int i = 0; i < 3; i++) {
        printf("Value %f: ", h_values[i]);
        for (int j = 0; j < n_bins; j++) {
            printf("%f ", h_probs[i * n_bins + j]);
        }
        printf("\n");
    }

   
    free(h_values);
    free(h_support);
    free(h_probs);
    cudaFree(d_values);
    cudaFree(d_support);
    cudaFree(d_probs);

    return 0;
}
