// main.cpp
#include <stdio.h>
#include "NaiveBayesTrain.cuh"

int main() {
    // Example Dataset: Each row is a sample with features, last column is the class label
    const int numSamples = 6;
    const int numFeatures = 2;
    const int numClasses = 2;
    const int numFeatureValues = 3; // Assuming features can take values 0, 1, 2

    int h_Dataset[numSamples][numFeatures + 1] = {
        {0, 1, 1}, // Feature 0=0, Feature 1=1, Class Label=1
        {1, 1, 1}, // Feature 0=1, Feature 1=1, Class Label=1
        {2, 2, 0}, // etc.
        {1, 0, 1},
        {0, 2, 0},
        {2, 1, 1}
    };

    int h_priors[numClasses] = {0};
    int h_likelihoods[numClasses * numFeatures * numFeatureValues] = {0};

    // Train the Naive Bayes model
    trainNaiveBayes(
        (int*)h_Dataset, h_priors, h_likelihoods,
        numSamples, numFeatures, numClasses, numFeatureValues
    );

    // Print priors
    printf("Priors:\n");
    for (int c = 0; c < numClasses; ++c) {
        printf("Class %d: %f\n", c, (float)h_priors[c] / numSamples);
    }

    // Print likelihoods
    printf("\nLikelihoods:\n");
    for (int c = 0; c < numClasses; ++c) {
        printf("Class %d:\n", c);
        for (int f = 0; f < numFeatures; ++f) {
            for (int v = 0; v < numFeatureValues; ++v) {
                int index = c * numFeatures * numFeatureValues + f * numFeatureValues + v;
                printf("Feature %d Value %d: %f\n", f, v, (float)h_likelihoods[index] / h_priors[c]);
            }
        }
        printf("\n");
    }

    return 0;
}
