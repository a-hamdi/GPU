// NaiveBayesTrain.cuh
#ifndef NAIVE_BAYES_TRAIN_CUH
#define NAIVE_BAYES_TRAIN_CUH

void trainNaiveBayes(
    int* h_Dataset, int* h_priors, int* h_likelihoods,
    int numSamples, int numFeatures, int numClasses, int numFeatureValues
);

#endif
