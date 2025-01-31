// NaiveBayes.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "NaiveBayesKernel.cuh"
#include "NaiveBayesTrain.cuh"

#define SHARED_SIZE 20

// CUDA Kernel to compute priors (P(Y = c)) and likelihoods (P(X | Y = c)).
__global__ void computePriorsAndLikelihood(
    int* d_Dataset, int* d_priors, int* d_likelihoods,
    int numSamples, int numFeatures, int numClasses, int numFeatureValues
) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int local_d_priors[SHARED_SIZE];
    __shared__ int local_d_likelihoods[SHARED_SIZE];

    // If the thread is within bounds
    if (threadId < numSamples) {
        // Each thread processes one data sample
        int classLabel = d_Dataset[threadId * (numFeatures + 1) + numFeatures]; // Class label is in the last column

        // Atomic update to calculate the prior
        atomicAdd(&local_d_priors[classLabel], 1);

        // Compute likelihood for each feature
        for (int fIdx = 0; fIdx < numFeatures; ++fIdx) {
            int featureValue = d_Dataset[threadId * (numFeatures + 1) + fIdx];
            int likelihoodIndex = classLabel * numFeatures * numFeatureValues + (fIdx * numFeatureValues) + featureValue;

            // Atomic update to the likelihood matrix
            atomicAdd(&local_d_likelihoods[likelihoodIndex], 1);
        }
    }

    // Synchronize threads before writing shared results back to global memory
    __syncthreads();

    // Write local results to global memory (only one thread needs to do this)
    if (threadIdx.x == 0) {
        for (int c = 0; c < numClasses; ++c) {
            atomicAdd(&d_priors[c], local_d_priors[c]);
        }

        for (int l = 0; l < numClasses * numFeatures * numFeatureValues; ++l) {
            atomicAdd(&d_likelihoods[l], local_d_likelihoods[l]);
        }
    }
}
