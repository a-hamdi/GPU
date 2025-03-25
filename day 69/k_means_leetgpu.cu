#include "solve.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>


__global__ void assign_and_accumulate(const float* data_x, const float* data_y,
                                      const float* centroid_x, const float* centroid_y,
                                      int* labels, float* sum_x, float* sum_y, int* count,
                                      int sample_size, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= sample_size) return;
    
    float x = data_x[idx];
    float y = data_y[idx];
    float minDist = 1e20f;
    int best = 0;
 
    for (int i = 0; i < k; i++) {
        float dx = x - centroid_x[i];
        float dy = y - centroid_y[i];
        float dist = dx * dx + dy * dy;
        if (dist < minDist) {
            minDist = dist;
            best = i;
        }
    }
    labels[idx] = best;

    atomicAdd(&sum_x[best], x);
    atomicAdd(&sum_y[best], y);
    atomicAdd(&count[best], 1);
}


__global__ void update_centroids(float* centroid_x, float* centroid_y,
                                 const float* sum_x, const float* sum_y, const int* count,
                                 int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    
   
    if (count[i] > 0) {
        centroid_x[i] = sum_x[i] / count[i];
        centroid_y[i] = sum_y[i] / count[i];
    }
}

void solve(const float* data_x, const float* data_y, int* labels,
           float* initial_centroid_x, float* initial_centroid_y,
           float* final_centroid_x, float* final_centroid_y,
           int sample_size, int k, int max_iterations) {

    
    float *d_sum_x, *d_sum_y;
    int *d_count;
    cudaMalloc(&d_sum_x, k * sizeof(float));
    cudaMalloc(&d_sum_y, k * sizeof(float));
    cudaMalloc(&d_count, k * sizeof(int));

    
    float *d_old_centroid_x, *d_old_centroid_y;
    cudaMalloc(&d_old_centroid_x, k * sizeof(float));
    cudaMalloc(&d_old_centroid_y, k * sizeof(float));

    
    float host_old_x[128], host_old_y[128], host_new_x[128], host_new_y[128];

    const float threshold = 0.0001f;

    
    int threadsPerBlock = 256;
    int blocksPerGrid = (sample_size + threadsPerBlock - 1) / threadsPerBlock;

    
    int threadsPerBlockCent = 256;
    int blocksPerGridCent = (k + threadsPerBlockCent - 1) / threadsPerBlockCent;

    for (int iter = 0; iter < max_iterations; iter++) {
        
        cudaMemset(d_sum_x, 0, k * sizeof(float));
        cudaMemset(d_sum_y, 0, k * sizeof(float));
        cudaMemset(d_count, 0, k * sizeof(int));

       
        assign_and_accumulate<<<blocksPerGrid, threadsPerBlock>>>(data_x, data_y,
                                                                  initial_centroid_x, initial_centroid_y,
                                                                  labels, d_sum_x, d_sum_y, d_count,
                                                                  sample_size, k);
        cudaDeviceSynchronize();

      
        cudaMemcpy(d_old_centroid_x, initial_centroid_x, k * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_old_centroid_y, initial_centroid_y, k * sizeof(float), cudaMemcpyDeviceToDevice);

        
        update_centroids<<<blocksPerGridCent, threadsPerBlockCent>>>(initial_centroid_x, initial_centroid_y,
                                                                      d_sum_x, d_sum_y, d_count, k);
        cudaDeviceSynchronize();

        
        cudaMemcpy(host_old_x, d_old_centroid_x, k * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_old_y, d_old_centroid_y, k * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_new_x, initial_centroid_x, k * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_new_y, initial_centroid_y, k * sizeof(float), cudaMemcpyDeviceToHost);

      
        bool converged = true;
        for (int i = 0; i < k; i++) {
            float dx = host_new_x[i] - host_old_x[i];
            float dy = host_new_y[i] - host_old_y[i];
            float distance = sqrtf(dx * dx + dy * dy);
            if (distance >= threshold) {
                converged = false;
                break;
            }
        }
        if (converged) {
            break;
        }
    }


    cudaMemcpy(final_centroid_x, initial_centroid_x, k * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(final_centroid_y, initial_centroid_y, k * sizeof(float), cudaMemcpyDeviceToDevice);


    cudaFree(d_sum_x);
    cudaFree(d_sum_y);
    cudaFree(d_count);
    cudaFree(d_old_centroid_x);
    cudaFree(d_old_centroid_y);
}
