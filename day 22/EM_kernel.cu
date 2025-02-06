#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Define number of clusters and data points
#define NUM_CLUSTERS 2     // number of clusters
#define N 1024             // number of data points
#define THREADS_PER_BLOCK 256

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// E-step kernel
__global__ void eStepKernel(float* data, int N, float* mu, float* sigma, 
                           float* pival, float* responsibilities) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        float x = data[idx];
        float probs[NUM_CLUSTERS];
        float sum = 0.0f;
        
        for (int k = 0; k < NUM_CLUSTERS; k++) {
            float diff = x - mu[k];
            float exponent = -0.5f * (diff * diff) / (sigma[k] * sigma[k]);
            float gauss = (1.0f / (sqrtf(2.0f * M_PI) * sigma[k])) * expf(exponent);
            probs[k] = pival[k] * gauss;
            sum += probs[k];
        }
        
        for (int k = 0; k < NUM_CLUSTERS; k++) {
            responsibilities[idx * NUM_CLUSTERS + k] = probs[k] / sum;
        }
    }
}

// M-step kernel
__global__ void mStepKernel(float* data, int N, float* responsibilities,
                           float* sum_gamma, float* sum_x, float* sum_x2) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        float x = data[idx];
        for (int k = 0; k < NUM_CLUSTERS; k++) {
            float gamma = responsibilities[idx * NUM_CLUSTERS + k];
            atomicAdd(&sum_gamma[k], gamma);
            atomicAdd(&sum_x[k], gamma * x);
            atomicAdd(&sum_x2[k], gamma * x * x);
        }
    }
}

int main() {
    // Seed the random number generator
    srand(static_cast<unsigned>(time(NULL)));

    // Generate synthetic 1D data
    float h_data[N];
    for (int i = 0; i < N; i++) {
        if (i < N/2) {
            h_data[i] = 2.0f + static_cast<float>(rand()) / RAND_MAX;
        } else {
            h_data[i] = 8.0f + static_cast<float>(rand()) / RAND_MAX;
        }
    }

    // Initial parameters (host)
    float h_mu[NUM_CLUSTERS] = {1.0f, 9.0f};           // means
    float h_sigma[NUM_CLUSTERS] = {1.0f, 1.0f};        // standard deviations
    float h_pival[NUM_CLUSTERS] = {0.5f, 0.5f};        // mixing coefficients

    // Allocate device memory
    float *d_data, *d_mu, *d_sigma, *d_pival;
    float *d_responsibilities, *d_sum_gamma, *d_sum_x, *d_sum_x2;

    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mu, NUM_CLUSTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sigma, NUM_CLUSTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pival, NUM_CLUSTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_responsibilities, N * NUM_CLUSTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum_gamma, NUM_CLUSTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum_x, NUM_CLUSTERS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum_x2, NUM_CLUSTERS * sizeof(float)));

    // Copy data and initial parameters to device
    CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mu, h_mu, NUM_CLUSTERS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sigma, h_sigma, NUM_CLUSTERS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pival, h_pival, NUM_CLUSTERS * sizeof(float), cudaMemcpyHostToDevice));

    // Calculate grid dimensions
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // Host arrays for M-step
    float h_sum_gamma[NUM_CLUSTERS];
    float h_sum_x[NUM_CLUSTERS];
    float h_sum_x2[NUM_CLUSTERS];

    // EM iterations
    int maxIter = 100;
    for (int iter = 0; iter < maxIter; iter++) {
        // E-step
        eStepKernel<<<blocks, THREADS_PER_BLOCK>>>(d_data, N, d_mu, d_sigma, 
                                                  d_pival, d_responsibilities);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Reset accumulators
        CUDA_CHECK(cudaMemset(d_sum_gamma, 0, NUM_CLUSTERS * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_sum_x, 0, NUM_CLUSTERS * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_sum_x2, 0, NUM_CLUSTERS * sizeof(float)));

        // M-step accumulation
        mStepKernel<<<blocks, THREADS_PER_BLOCK>>>(d_data, N, d_responsibilities,
                                                  d_sum_gamma, d_sum_x, d_sum_x2);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy results back to host
        CUDA_CHECK(cudaMemcpy(h_sum_gamma, d_sum_gamma, NUM_CLUSTERS * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_sum_x, d_sum_x, NUM_CLUSTERS * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_sum_x2, d_sum_x2, NUM_CLUSTERS * sizeof(float), cudaMemcpyDeviceToHost));

        // Update parameters
        for (int k = 0; k < NUM_CLUSTERS; k++) {
            if (h_sum_gamma[k] > 1e-6f) {
                // Update mean
                h_mu[k] = h_sum_x[k] / h_sum_gamma[k];
                
                // Update variance and standard deviation
                float variance = h_sum_x2[k] / h_sum_gamma[k] - h_mu[k] * h_mu[k];
                h_sigma[k] = sqrtf(fmax(variance, 1e-6f));
                
                // Update mixing coefficient
                h_pival[k] = h_sum_gamma[k] / N;
            }
        }

        // Copy updated parameters back to device
        CUDA_CHECK(cudaMemcpy(d_mu, h_mu, NUM_CLUSTERS * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sigma, h_sigma, NUM_CLUSTERS * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pival, h_pival, NUM_CLUSTERS * sizeof(float), cudaMemcpyHostToDevice));

        // Print current parameters
        if (iter % 10 == 0) {  // Print every 10 iterations
            std::cout << "Iteration " << iter << ":\n";
            for (int k = 0; k < NUM_CLUSTERS; k++) {
                std::cout << "Cluster " << k << ": "
                         << "mu = " << h_mu[k] << ", "
                         << "sigma = " << h_sigma[k] << ", "
                         << "pi = " << h_pival[k] << std::endl;
            }
            std::cout << std::endl;
        }
    }

    // Free device memory
    cudaFree(d_data);
    cudaFree(d_mu);
    cudaFree(d_sigma);
    cudaFree(d_pival);
    cudaFree(d_responsibilities);
    cudaFree(d_sum_gamma);
    cudaFree(d_sum_x);
    cudaFree(d_sum_x2);

    return 0;
}
