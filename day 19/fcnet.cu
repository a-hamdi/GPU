// use this command: nvcc -o fcnet fcnet.cu -lcudnn -lcublas -lcurand

#include <cudnn.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cstdlib>
#include <ctime>
#include <iostream>

#define CHECK_CUDA(func) { \
    cudaError_t status = (func); \
    if (status != cudaSuccess) { \
        std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(status) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUDNN(func) { \
    cudnnStatus_t status = (func); \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "cuDNN Error at " << __FILE__ << ":" << __LINE__ << " - " << cudnnGetErrorString(status) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// Network parameters
const int input_size = 1000;
const int hidden_size = 512;
const int output_size = 10;
const int batch_size = 64;
const float learning_rate = 0.001f;
const int epochs = 10;

// Helper function to initialize weights
void initialize_weights(float* weights, int size) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
    curandGenerateUniform(gen, weights, size);
    curandDestroyGenerator(gen);
}

int main() {
    // Initialize CUDA and cuDNN
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    // Create tensor descriptors
    cudnnTensorDescriptor_t input_desc, hidden1_desc, hidden2_desc, output_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&hidden1_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&hidden2_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
    
    // Set tensor dimensions (NCHW format)
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                          batch_size, input_size, 1, 1));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(hidden1_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                          batch_size, hidden_size, 1, 1));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(hidden2_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                          batch_size, hidden_size, 1, 1));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                          batch_size, output_size, 1, 1));

    // Create filter descriptors (weights)
    cudnnFilterDescriptor_t fc1_w_desc, fc2_w_desc, fc3_w_desc;
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&fc1_w_desc));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(fc1_w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                          hidden_size, input_size, 1, 1));
    
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&fc2_w_desc));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(fc2_w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                          hidden_size, hidden_size, 1, 1));
    
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&fc3_w_desc));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(fc3_w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                          output_size, hidden_size, 1, 1));

    // Create activation descriptor (ReLU)
    cudnnActivationDescriptor_t relu_desc;
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&relu_desc));
    CHECK_CUDNN(cudnnSetActivationDescriptor(relu_desc, CUDNN_ACTIVATION_RELU, 
                                             CUDNN_NOT_PROPAGATE_NAN, 0.0));

    // Allocate device memory
    float *d_input, *d_labels, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, batch_size * input_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_labels, batch_size * output_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, batch_size * output_size * sizeof(float)));

    // Initialize weights and biases
    float *d_w1, *d_b1, *d_w2, *d_b2, *d_w3, *d_b3;
    CHECK_CUDA(cudaMalloc(&d_w1, hidden_size * input_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b1, hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w2, hidden_size * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b2, hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w3, output_size * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b3, output_size * sizeof(float)));

    initialize_weights(d_w1, hidden_size * input_size);
    initialize_weights(d_w2, hidden_size * hidden_size);
    initialize_weights(d_w3, output_size * hidden_size);
    CHECK_CUDA(cudaMemset(d_b1, 0, hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_b2, 0, hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_b3, 0, output_size * sizeof(float)));

    // Generate dummy data
    initialize_weights(d_input, batch_size * input_size);
    initialize_weights(d_labels, batch_size * output_size);

    // Create convolution descriptors
    cudnnConvolutionDescriptor_t conv_desc;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1,
                                               CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Forward pass
        float alpha = 1.0f, beta = 0.0f;
        float* d_hidden1, *d_hidden2;
        CHECK_CUDA(cudaMalloc(&d_hidden1, batch_size * hidden_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_hidden2, batch_size * hidden_size * sizeof(float)));

        // Layer 1: Input -> Hidden1
        CHECK_CUDNN(cudnnConvolutionForward(cudnn, &alpha,
                                           input_desc, d_input,
                                           fc1_w_desc, d_w1,
                                           conv_desc,
                                           CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                                           nullptr, 0,
                                           &beta,
                                           hidden1_desc, d_hidden1));
        CHECK_CUDNN(cudnnAddTensor(cudnn, &alpha, hidden1_desc, d_b1,
                                  &alpha, hidden1_desc, d_hidden1));
        CHECK_CUDNN(cudnnActivationForward(cudnn, relu_desc,
                                          &alpha, hidden1_desc, d_hidden1,
                                          &beta, hidden1_desc, d_hidden1));

        // Layer 2: Hidden1 -> Hidden2
        CHECK_CUDNN(cudnnConvolutionForward(cudnn, &alpha,
                                           hidden1_desc, d_hidden1,
                                           fc2_w_desc, d_w2,
                                           conv_desc,
                                           CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                                           nullptr, 0,
                                           &beta,
                                           hidden2_desc, d_hidden2));
        CHECK_CUDNN(cudnnAddTensor(cudnn, &alpha, hidden2_desc, d_b2,
                                  &alpha, hidden2_desc, d_hidden2));
        CHECK_CUDNN(cudnnActivationForward(cudnn, relu_desc,
                                          &alpha, hidden2_desc, d_hidden2,
                                          &beta, hidden2_desc, d_hidden2));

        // Layer 3: Hidden2 -> Output
        CHECK_CUDNN(cudnnConvolutionForward(cudnn, &alpha,
                                           hidden2_desc, d_hidden2,
                                           fc3_w_desc, d_w3,
                                           conv_desc,
                                           CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                                           nullptr, 0,
                                           &beta,
                                           output_desc, d_output));
        CHECK_CUDNN(cudnnAddTensor(cudnn, &alpha, output_desc, d_b3,
                                  &alpha, output_desc, d_output));

        // Cleanup intermediate buffers
        CHECK_CUDA(cudaFree(d_hidden1));
        CHECK_CUDA(cudaFree(d_hidden2));

        // Inference (example)
        if (epoch == epochs - 1) {
            float* h_output = new float[batch_size * output_size];
            CHECK_CUDA(cudaMemcpy(h_output, d_output, batch_size * output_size * sizeof(float),
                                 cudaMemcpyDeviceToHost));
            std::cout << "Final inference results sample: " << h_output[0] << std::endl;
            delete[] h_output;
        }
    }

    // Cleanup
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_labels));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_w1));
    CHECK_CUDA(cudaFree(d_b1));
    CHECK_CUDA(cudaFree(d_w2));
    CHECK_CUDA(cudaFree(d_b2));
    CHECK_CUDA(cudaFree(d_w3));
    CHECK_CUDA(cudaFree(d_b3));

    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));
    CHECK_CUDNN(cudnnDestroyActivationDescriptor(relu_desc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(fc1_w_desc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(fc2_w_desc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(fc3_w_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(hidden1_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(hidden2_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_desc));
    CHECK_CUDNN(cudnnDestroy(cudnn));

    return 0;
}

