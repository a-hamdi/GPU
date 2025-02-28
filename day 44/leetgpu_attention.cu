#include "solve.h"
#include <cuda_runtime.h>
#include <math.h>


__global__ void compute_scores(const float* Q, const float* K, float* scores,
                               int N, int d_model, int d_k) {
    int head = blockIdx.z;  
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N && j < N) {
        float sum = 0.0f;
        int offset = head * d_k;
        for (int k = 0; k < d_k; k++) {
            float q_val = Q[i * d_model + offset + k];
            float k_val = K[j * d_model + offset + k];
            sum += q_val * k_val;
        }
        // Scale the dot product by sqrt(d_k)
        float scale = 1.0f / sqrtf((float)d_k);
        // Store into scores array (each head gets a contiguous N*N block)
        scores[ head * N * N + i * N + j ] = sum * scale;
    }
}


__global__ void softmax_kernel(float* scores, int N) {
    int head = blockIdx.z;  
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < N) {
        int base = head * N * N + row * N;
        
        float max_val = -1e20f;
        for (int j = 0; j < N; j++) {
            float val = scores[ base + j ];
            if(val > max_val)
                max_val = val;
        }
        
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            float exp_val = expf(scores[ base + j ] - max_val);
            scores[ base + j ] = exp_val; 
            sum += exp_val;
        }
        
        for (int j = 0; j < N; j++) {
            scores[ base + j ] /= sum;
        }
    }
}


__global__ void compute_output_kernel(const float* scores, const float* V, float* out,
                                        int N, int d_model, int d_k) {
    int head = blockIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x; // k in [0, d_k)
    if (i < N && k < d_k) {
        float sum = 0.0f;
        int offset = head * d_k;
        for (int j = 0; j < N; j++) {
            float attn = scores[ head * N * N + i * N + j ];
            float v_val = V[j * d_model + offset + k];
            sum += attn * v_val;
        }
       
        out[i * d_model + offset + k] = sum;
    }
}

void solve(const float* Q, const float* K, const float* V,
           float* output, int N, int d_model, int h) {
    
    int d_k = d_model / h;
    
    
    float *d_Q, *d_K, *d_V, *d_scores, *d_output;
    size_t modelBytes = N * d_model * sizeof(float);
    size_t scoreBytes = h * N * N * sizeof(float);
    
    cudaMalloc((void**)&d_Q, modelBytes);
    cudaMalloc((void**)&d_K, modelBytes);
    cudaMalloc((void**)&d_V, modelBytes);
    cudaMalloc((void**)&d_scores, scoreBytes);
    cudaMalloc((void**)&d_output, modelBytes);
    
   
    cudaMemcpy(d_Q, Q, modelBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, modelBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, modelBytes, cudaMemcpyHostToDevice);
    

    dim3 blockDim1(16, 16);
    dim3 gridDim1((N + blockDim1.x - 1) / blockDim1.x,
                  (N + blockDim1.y - 1) / blockDim1.y,
                  h);
    compute_scores<<<gridDim1, blockDim1>>>(d_Q, d_K, d_scores, N, d_model, d_k);
    cudaDeviceSynchronize();

    dim3 blockDim2(1, 256); 
    dim3 gridDim2(1, (N + blockDim2.y - 1) / blockDim2.y, h);
    softmax_kernel<<<gridDim2, blockDim2>>>(d_scores, N);
    cudaDeviceSynchronize();
    
    dim3 blockDim3(16, 16);
    dim3 gridDim3((d_k + blockDim3.x - 1) / blockDim3.x,
                  (N + blockDim3.y - 1) / blockDim3.y,
                  h);
    compute_output_kernel<<<gridDim3, blockDim3>>>(d_scores, d_V, d_output, N, d_model, d_k);
    cudaDeviceSynchronize();
    
    
    cudaMemcpy(output, d_output, modelBytes, cudaMemcpyDeviceToHost);
    
    
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_scores);
    cudaFree(d_output);
}
