#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

#define HIDDEN_SIZE 128
#define INPUT_SIZE 128
#define SEQ_LEN 50
#define BATCH_SIZE 32

// Sigmoid activation function
__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Tanh activation function
__device__ float tanh_activation(float x) {
    return tanhf(x);
}

// LSTM kernel for a single timestep
__global__ void lstm_forward(float* input, float* h_prev, float* c_prev, float* W, float* U, float* b, float* h_out, float* c_out) {
    int batch_idx = blockIdx.x;
    int neuron_idx = threadIdx.x;
    
    if (neuron_idx >= HIDDEN_SIZE) return; // Ensure valid index range
    
    float x_t = input[batch_idx * INPUT_SIZE + neuron_idx];
    float h_prev_t = h_prev[batch_idx * HIDDEN_SIZE + neuron_idx];
    float c_prev_t = c_prev[batch_idx * HIDDEN_SIZE + neuron_idx];
    
    float i_t = sigmoid(x_t * W[neuron_idx] + h_prev_t * U[neuron_idx] + b[neuron_idx]);
    float f_t = sigmoid(x_t * W[neuron_idx + HIDDEN_SIZE] + h_prev_t * U[neuron_idx + HIDDEN_SIZE] + b[neuron_idx + HIDDEN_SIZE]);
    float o_t = sigmoid(x_t * W[neuron_idx + 2 * HIDDEN_SIZE] + h_prev_t * U[neuron_idx + 2 * HIDDEN_SIZE] + b[neuron_idx + 2 * HIDDEN_SIZE]);
    float g_t = tanh_activation(x_t * W[neuron_idx + 3 * HIDDEN_SIZE] + h_prev_t * U[neuron_idx + 3 * HIDDEN_SIZE] + b[neuron_idx + 3 * HIDDEN_SIZE]);
    
    float c_t = f_t * c_prev_t + i_t * g_t;
    float h_t = o_t * tanh_activation(c_t);
    
    h_out[batch_idx * HIDDEN_SIZE + neuron_idx] = h_t;
    c_out[batch_idx * HIDDEN_SIZE + neuron_idx] = c_t;
}

// Function to launch bidirectional LSTM
void bidirectional_lstm(float* input, float* h_forward, float* c_forward, float* h_backward, float* c_backward, float* W, float* U, float* b, float* output) {
    for (int t = 0; t < SEQ_LEN; t++) {
        lstm_forward<<<BATCH_SIZE, HIDDEN_SIZE>>>(input + t * BATCH_SIZE * INPUT_SIZE, h_forward, c_forward, W, U, b, h_forward, c_forward);
        lstm_forward<<<BATCH_SIZE, HIDDEN_SIZE>>>(input + (SEQ_LEN - 1 - t) * BATCH_SIZE * INPUT_SIZE, h_backward, c_backward, W, U, b, h_backward, c_backward);
    }
    cudaDeviceSynchronize();
    
    // Concatenate forward and backward outputs
    cudaMemcpy(output, h_forward, BATCH_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(output + BATCH_SIZE * HIDDEN_SIZE, h_backward, BATCH_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
}

int main() {
    srand(time(NULL));
    // Allocate host memory
    std::vector<float> h_input(SEQ_LEN * BATCH_SIZE * INPUT_SIZE, 1.0f);
    std::vector<float> h_output(2 * BATCH_SIZE * HIDDEN_SIZE, 0.0f);
    std::vector<float> h_W(4 * HIDDEN_SIZE * INPUT_SIZE);
    std::vector<float> h_U(4 * HIDDEN_SIZE * HIDDEN_SIZE);
    std::vector<float> h_b(4 * HIDDEN_SIZE);
    
    for (auto& w : h_W) w = ((float) rand() / RAND_MAX) * 0.1f;
    for (auto& u : h_U) u = ((float) rand() / RAND_MAX) * 0.1f;
    for (auto& b : h_b) b = 0.0f;
    
    // Allocate device memory
    float *d_input, *d_h_forward, *d_c_forward, *d_h_backward, *d_c_backward, *d_W, *d_U, *d_b, *d_output;
    cudaMalloc(&d_input, SEQ_LEN * BATCH_SIZE * INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_h_forward, BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_c_forward, BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_h_backward, BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_c_backward, BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_W, 4 * HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_U, 4 * HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_b, 4 * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_output, 2 * BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
    
    // Initialize memory on device
    cudaMemcpy(d_input, h_input.data(), SEQ_LEN * BATCH_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, h_W.data(), 4 * HIDDEN_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_U, h_U.data(), 4 * HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), 4 * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_h_forward, 0, BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMemset(d_c_forward, 0, BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMemset(d_h_backward, 0, BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMemset(d_c_backward, 0, BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
    
    // Run bidirectional LSTM
    bidirectional_lstm(d_input, d_h_forward, d_c_forward, d_h_backward, d_c_backward, d_W, d_U, d_b, d_output);
    
    // Copy result back to host
    cudaMemcpy(h_output.data(), d_output, 2 * BATCH_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print output
    std::cout << "Bidirectional LSTM Output:\n";
    for (int i = 0; i < 10; i++) { 
        std::cout << h_output[i] << " ";
    }
    std::cout << "...\n";
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_h_forward);
    cudaFree(d_c_forward);
    cudaFree(d_h_backward);
    cudaFree(d_c_backward);
    cudaFree(d_W);
    cudaFree(d_U);
    cudaFree(d_b);
    cudaFree(d_output);
    
    return 0;
}