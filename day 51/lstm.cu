#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <cmath>

#define TIMESTEPS 5

// Device function: Sigmoid activation
__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// LSTM kernel for one timestep
__global__ void lstm_kernel(const float* x_t, const float* h_t_prev, const float* C_t_prev,
                            const float* W, const float* U, const float* b,
                            float* h_t, float* C_t, int hidden_size, int batch_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= hidden_size * batch_size) return;

    int neuron_idx = idx % hidden_size;  // neuron index within the hidden layer

    // Each gate's parameters are stored sequentially in W, U, and b:
    // Gate 0 (Input): index [0, hidden_size)
    // Gate 1 (Forget): index [hidden_size, 2*hidden_size)
    // Gate 2 (Output): index [2*hidden_size, 3*hidden_size)
    // Gate 3 (Candidate): index [3*hidden_size, 4*hidden_size)
    float input_gate = sigmoid(W[neuron_idx] * x_t[idx] +
                               U[neuron_idx] * h_t_prev[idx] +
                               b[neuron_idx]);

    float forget_gate = sigmoid(W[hidden_size + neuron_idx] * x_t[idx] +
                                U[hidden_size + neuron_idx] * h_t_prev[idx] +
                                b[hidden_size + neuron_idx]);

    float output_gate = sigmoid(W[2 * hidden_size + neuron_idx] * x_t[idx] +
                                U[2 * hidden_size + neuron_idx] * h_t_prev[idx] +
                                b[2 * hidden_size + neuron_idx]);

    float candidate = tanhf(W[3 * hidden_size + neuron_idx] * x_t[idx] +
                            U[3 * hidden_size + neuron_idx] * h_t_prev[idx] +
                            b[3 * hidden_size + neuron_idx]);

    // Update cell state and hidden state
    C_t[idx] = forget_gate * C_t_prev[idx] + input_gate * candidate;
    h_t[idx] = output_gate * tanhf(C_t[idx]);
}

int main() {
    int hidden_size = 128;
    int batch_size = 1;  // Single batch for simplicity
    int num_elements = hidden_size * batch_size;

    // Allocate unified memory for inputs, states, weights, and biases
    float *x_t, *h_t_prev, *C_t_prev, *W, *U, *b, *h_t, *C_t;
    cudaMallocManaged(&x_t, num_elements * sizeof(float));
    cudaMallocManaged(&h_t_prev, num_elements * sizeof(float));
    cudaMallocManaged(&C_t_prev, num_elements * sizeof(float));
    cudaMallocManaged(&W, 4 * hidden_size * sizeof(float));
    cudaMallocManaged(&U, 4 * hidden_size * sizeof(float));
    cudaMallocManaged(&b, 4 * hidden_size * sizeof(float));
    cudaMallocManaged(&h_t, num_elements * sizeof(float));
    cudaMallocManaged(&C_t, num_elements * sizeof(float));

    // Initialize previous hidden and cell states to nonzero values
    for (int i = 0; i < num_elements; i++) {
        h_t_prev[i] = 0.5f;
        C_t_prev[i] = 0.5f;
        x_t[i] = 1.0f;  // Constant input for simplicity
    }

    // Initialize weights and biases for each gate
    for (int i = 0; i < hidden_size; i++) {
        // Input gate
        W[i] = 0.5f;
        U[i] = 0.5f;
        b[i] = 0.1f;

        // Forget gate
        W[hidden_size + i] = 0.5f;
        U[hidden_size + i] = 0.5f;
        b[hidden_size + i] = 0.2f;

        // Output gate
        W[2 * hidden_size + i] = 0.5f;
        U[2 * hidden_size + i] = 0.5f;
        b[2 * hidden_size + i] = 0.3f;

        // Candidate gate
        W[3 * hidden_size + i] = 0.5f;
        U[3 * hidden_size + i] = 0.5f;
        b[3 * hidden_size + i] = 0.0f;
    }

    int threads_per_block = 128;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "LSTM processing over " << TIMESTEPS << " timesteps:" << std::endl;

    // Process multiple timesteps
    for (int t = 0; t < TIMESTEPS; t++) {
        lstm_kernel<<<num_blocks, threads_per_block>>>(x_t, h_t_prev, C_t_prev,
                                                         W, U, b, h_t, C_t,
                                                         hidden_size, batch_size);
        cudaDeviceSynchronize();

        // For next timestep, update previous states with current results
        for (int i = 0; i < num_elements; i++) {
            h_t_prev[i] = h_t[i];
            C_t_prev[i] = C_t[i];
        }

        // Print the state of the first neuron after this timestep
        std::cout << "After timestep " << t + 1 << ": h_t[0] = " << h_t[0]
                  << ", C_t[0] = " << C_t[0] << std::endl;
    }

    // Free allocated memory
    cudaFree(x_t);
    cudaFree(h_t_prev);
    cudaFree(C_t_prev);
    cudaFree(W);
    cudaFree(U);
    cudaFree(b);
    cudaFree(h_t);
    cudaFree(C_t);

    return 0;
}
