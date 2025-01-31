// NaiveBayesTrain.cpp
#include <cuda_runtime.h>
#include "NaiveBayesTrain.cuh"
#include "NaiveBayesKernel.cuh"

void trainNaiveBayes(
    int* h_Dataset, int* h_priors, int* h_likelihoods,
    int numSamples, int numFeatures, int numClasses, int numFeatureValues
) {
    // Device pointers
    int* d_Dataset;
    int* d_priors;
    int* d_likelihoods;

    // Allocate memory on the GPU
    int datasetSize = numSamples * (numFeatures + 1) * sizeof(int); // +1 for the class label
    int priorsSize = numClasses * sizeof(int);
    int likelihoodsSize = numClasses * numFeatures * numFeatureValues * sizeof(int);

    cudaMalloc((void**)&d_Dataset, datasetSize);
    cudaMalloc((void**)&d_priors, priorsSize);
    cudaMalloc((void**)&d_likelihoods, likelihoodsSize);

    // Copy data from host to device
    cudaMemcpy(d_Dataset, h_Dataset, datasetSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_priors, h_priors, priorsSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_likelihoods, h_likelihoods, likelihoodsSize, cudaMemcpyHostToDevice);

    // Number of threads and blocks
    int threadsPerBlock = 256;
    int numBlocks = (numSamples + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the CUDA kernel
    computePriorsAndLikelihood<<<numBlocks, threadsPerBlock>>>( 
        d_Dataset, d_priors, d_likelihoods,
        numSamples, numFeatures, numClasses, numFeatureValues
    );

    // Copy results back from device to host
    cudaMemcpy(h_priors, d_priors, priorsSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_likelihoods, d_likelihoods, likelihoodsSize, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_Dataset);
    cudaFree(d_priors);
    cudaFree(d_likelihoods);
}
