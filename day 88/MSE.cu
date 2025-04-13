#include <cuda_runtime.h>
#include <cstdio>

__global__ void mseKernel(const float* predictions, const float* targets, size_t numElements, float* sum) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        float diff = predictions[idx] - targets[idx];
        float sq_diff = diff * diff;
        
        atomicAdd(sum, sq_diff);
    }
}

extern "C" void solution(const float* predictions, const float* targets, float* output, size_t* shape, size_t ndim) {

    size_t* hostShape = new size_t[ndim];
    cudaMemcpy(hostShape, shape, ndim * sizeof(size_t), cudaMemcpyDeviceToHost);

    size_t numElements = 1;
    for (size_t i = 0; i < ndim; i++) {
        numElements *= hostShape[i];
    }
    delete[] hostShape;


    float init = 0.0f;
    cudaMemcpy(output, &init, sizeof(float), cudaMemcpyHostToDevice);


    int threadsPerBlock = 256;
    int blocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    mseKernel<<<blocks, threadsPerBlock>>>(predictions, targets, numElements, output);
    cudaDeviceSynchronize(); 

    float hostSum = 0.0f;
    cudaMemcpy(&hostSum, output, sizeof(float), cudaMemcpyDeviceToHost);

    float mse = hostSum / numElements;

    cudaMemcpy(output, &mse, sizeof(float), cudaMemcpyHostToDevice);
}
