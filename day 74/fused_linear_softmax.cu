#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCK_SIZE 256

__global__ void LinearKernel(float* input, float* weights, float* bias, float* output, int in_features, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < out_features) {
        float sum = bias[idx];
        for (int i = 0; i < in_features; i++) {
            sum += input[i] * weights[i * out_features + idx];
        }
        output[idx] = sum;
    }
}

__global__ void softmaxCrossEntropyKernel(float* logits, int* labels, float* loss, int num_classes) {
    extern __shared__ float exp_sums[];
    int idx = threadIdx.x;

    float max_val = -INFINITY;
    for (int i = 0; i < num_classes; i++) {
        max_val = fmaxf(max_val, logits[i]);
    }
    float sum = 0.0;
    for (int i = 0; i < num_classes; i++) {
        exp_sums[i] = expf(logits[i] - max_val);
        sum += exp_sums[i];
    }
    float log_prob = logf(exp_sums[labels[0]] / sum);
    *loss = -log_prob;
}

void runFusedOperations(float* input, float* weights, float* bias, int* label, int in_features, int out_features) {
    float *d_input, *d_weights, *d_bias, *d_output, *d_loss;
    int *d_label;

    cudaMalloc((void**)&d_input, in_features * sizeof(float));
    cudaMalloc((void**)&d_weights, in_features * out_features * sizeof(float));
    cudaMalloc((void**)&d_bias, out_features * sizeof(float));
    cudaMalloc((void**)&d_output, out_features * sizeof(float));
    cudaMalloc((void**)&d_label, sizeof(int));
    cudaMalloc((void**)&d_loss, sizeof(float));

    cudaMemcpy(d_input, input, in_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, in_features * out_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, out_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_label, label, sizeof(int), cudaMemcpyHostToDevice);

    LinearKernel<<<(out_features + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_input, d_weights, d_bias, d_output, in_features, out_features);
    softmaxCrossEntropyKernel<<<1, out_features, out_features * sizeof(float)>>>(d_output, d_label, d_loss, out_features);

    float loss;
    cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Cross-entropy loss: %f\n", loss);

    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_bias);
    cudaFree(d_output);
    cudaFree(d_label);
    cudaFree(d_loss);
}

int main() {
    int in_features = 3;
    int out_features = 2;
    float input[3] = {1.0, 2.0, 3.0};
    float weights[6] = {0.2, 0.8, -0.5, 1.0, -1.2, 0.3};
    float bias[2] = {0.1, -0.2};
    int label = 1;

    runFusedOperations(input, weights, bias, &label, in_features, out_features);
    return 0;
}
