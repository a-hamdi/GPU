// NaiveBayesKernel.cuh
#ifndef NAIVE_BAYES_KERNEL_CUH
#define NAIVE_BAYES_KERNEL_CUH

__global__ void computePriorsAndLikelihood(
    int* d_Dataset, int* d_priors, int* d_likelihoods,
    int numSamples, int numFeatures, int numClasses, int numFeatureValues
);

#endif
