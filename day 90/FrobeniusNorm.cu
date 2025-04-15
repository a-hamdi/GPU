#include <cuda_runtime.h>


__global__ void calculateSumOfSquares(const float* X, float* partialSums, size_t size) {
    extern __shared__ float sharedData[];
    
   
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    

    sharedData[tid] = 0.0f;
    
    
    while (i < size) {
        sharedData[tid] += X[i] * X[i];
        i += blockDim.x * gridDim.x;
    }
    
    
    __syncthreads();
    
   
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedData[tid] += sharedData[tid + s];
        }
        __syncthreads();
    }
    
    
    if (tid == 0) {
        partialSums[blockIdx.x] = sharedData[0];
    }
}


__global__ void normalizeByFrobeniusNorm(const float* X, float* Y, size_t size, float frobeniusNorm) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < size) {
        Y[i] = X[i] / frobeniusNorm;
    }
}

extern "C" void solution(const float* X, float* Y, size_t size) {
    
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    int maxBlocks = 1024; 
    
    if (gridSize > maxBlocks) {
        gridSize = maxBlocks;
    }
    
    float* d_partialSums;
    cudaMalloc(&d_partialSums, gridSize * sizeof(float));
    
    calculateSumOfSquares<<<gridSize, blockSize, blockSize * sizeof(float)>>>(X, d_partialSums, size);
    
    float* h_partialSums = new float[gridSize];
    cudaMemcpy(h_partialSums, d_partialSums, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
    
    float sumOfSquares = 0.0f;
    for (int i = 0; i < gridSize; i++) {
        sumOfSquares += h_partialSums[i];
    }
    
    float frobeniusNorm = sqrt(sumOfSquares);
    
    if (frobeniusNorm < 1e-10) {
        frobeniusNorm = 1.0f; 
    }
    
    normalizeByFrobeniusNorm<<<(size + blockSize - 1) / blockSize, blockSize>>>(X, Y, size, frobeniusNorm);
    
    delete[] h_partialSums;
    cudaFree(d_partialSums);
}