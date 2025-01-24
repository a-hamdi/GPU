#include <iostream>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>

#define SRAM_SIZE 1024                    // M: SRAM size
#define sequence_length 2              // N: sequence length
#define embed_dimension 2              // d: embedding dimension

// Define constant sizes to be used for block sizes
constexpr int Block_column_size = SRAM_SIZE / (4 * embed_dimension); // Bc
constexpr int Block_row_size = std::min(SRAM_SIZE / (4 * embed_dimension), embed_dimension); // Br

// Ensure we don't have a division by zero situation
static_assert(Block_column_size > 0, "Block_column_size must be greater than 0");
static_assert(Block_row_size > 0, "Block_row_size must be greater than 0");

constexpr int Total_row_blocks = (sequence_length + Block_row_size - 1) / Block_row_size; // Tr
constexpr int Total_column_blocks = (sequence_length + Block_column_size - 1) / Block_column_size; // Tc

__global__ void flashAttentionForward(
    const float *Query,                 // Q
    const float *Key,                   // K
    const float *Value,                 // V
    float *Output,                      // O
    float *max_values,                  // m
    float *sum_values,                  // l
    const float attention_scale)        // 1/sqrt(d)
{
    int thread_idx = threadIdx.x;

    // Thread-local storage for attention scores and weights
    float attention_scores[Block_row_size * Block_column_size];
    float attention_weights[Block_row_size * Block_column_size];

    // Shared memory blocks
    float Query_block[Block_row_size * embed_dimension];
    float Key_block[Block_column_size * embed_dimension];
    float Value_block[Block_column_size * embed_dimension];

    for (int col_block = 0; col_block < Total_column_blocks; ++col_block)
    {
        // Load Key and Value blocks from global memory
        if (thread_idx < Block_column_size) {
            for (int d = 0; d < embed_dimension; ++d) {
                Key_block[thread_idx * embed_dimension + d] = 
                    Key[col_block * Block_column_size * embed_dimension + thread_idx * embed_dimension + d];
                Value_block[thread_idx * embed_dimension + d] = 
                    Value[col_block * Block_column_size * embed_dimension + thread_idx * embed_dimension + d];
            }
        }
        __syncthreads();

        for (int row_block = 0; row_block < Total_row_blocks; ++row_block)
        {
            if (thread_idx < Block_row_size) {
                // Load Query block
                for (int d = 0; d < embed_dimension; ++d) {
                    Query_block[thread_idx * embed_dimension + d] = 
                        Query[row_block * Block_row_size * embed_dimension + thread_idx * embed_dimension + d];
                }
            }
            __syncthreads();

            // Compute attention scores for this row
            if (thread_idx < Block_row_size) {
                float row_max = -1e20;  // Use a large negative float
                for (int k = 0; k < Block_column_size; ++k) {
                    float score = 0.0f;
                    for (int d = 0; d < embed_dimension; ++d) {
                        score += Query_block[thread_idx * embed_dimension + d] * 
                                Key_block[k * embed_dimension + d];
                    }
                    score *= attention_scale;
                    attention_scores[thread_idx * Block_column_size + k] = score;
                    row_max = fmaxf(row_max, score);
                }

                // Compute attention weights with softmax
                float row_sum = 0.0f;
                for (int k = 0; k < Block_column_size; ++k) {
                    float weight = expf(attention_scores[thread_idx * Block_column_size + k] - row_max);
                    attention_weights[thread_idx * Block_column_size + k] = weight;
                    row_sum += weight;
                }

                // Update output
                for (int d = 0; d < embed_dimension; ++d) {
                    float weighted_sum = 0.0f;
                    for (int k = 0; k < Block_column_size; ++k) {
                        weighted_sum += attention_weights[thread_idx * Block_column_size + k] * 
                                      Value_block[k * embed_dimension + d];
                    }
                    Output[row_block * Block_row_size * embed_dimension + thread_idx * embed_dimension + d] = 
                        (row_sum > 0) ? (weighted_sum / row_sum) : 0; // Avoid division by zero
                }
            }
            __syncthreads();
        }
    }
}

int main()
{
    // Host memory allocation
    float (*Query)[embed_dimension] = new float[sequence_length][embed_dimension];
    float (*Key)[embed_dimension] = new float[sequence_length][embed_dimension];
    float (*Value)[embed_dimension] = new float[sequence_length][embed_dimension];
    float (*Output)[embed_dimension] = new float[sequence_length][embed_dimension];
    float *sum_values = new float[sequence_length]();  // Initialize to zeros
    float *max_values = new float[sequence_length];

    // Initialize max_values to a very small negative number
    for (int i = 0; i < sequence_length; i++) {
        max_values[i] = -1e20;  // Large negative float
    }

    // Initialization (random values between -1 and 1): 
    for (int i = 0; i < sequence_length; i++) {
        for (int j = 0; j < embed_dimension; j++) {
            Query[i][j] = 2.0f * rand() / RAND_MAX - 1.0f;
            Key[i][j] = 2.0f * rand() / RAND_MAX - 1.0f;
            Value[i][j] = 2.0f * rand() / RAND_MAX - 1.0f;
            Output[i][j] = 0.0f;  // Initialize output to zeros
        }
    }

    // Device memory pointers
    float *device_Query, *device_Key, *device_Value, *device_Output;
    float *device_max_values, *device_sum_values;

    // Allocate device memory
    cudaMalloc(&device_Query, sequence_length * embed_dimension * sizeof(float));
    cudaMalloc(&device_Key, sequence_length * embed_dimension * sizeof(float));
    cudaMalloc(&device_Value, sequence_length * embed_dimension * sizeof(float));
    cudaMalloc(&device_Output, sequence_length * embed_dimension * sizeof(float));
    cudaMalloc(&device_sum_values, sequence_length * sizeof(float));
    cudaMalloc(&device_max_values, sequence_length * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(device_Query, Query, 
               sequence_length * embed_dimension * sizeof(float), 
               cudaMemcpyHostToDevice);
    cudaMemcpy(device_Key, Key, 
               sequence_length * embed_dimension * sizeof(float), 
               cudaMemcpyHostToDevice);
    cudaMemcpy(device_Value, Value, 
               sequence_length * embed_dimension * sizeof(float), 
               cudaMemcpyHostToDevice);
    cudaMemcpy(device_Output, Output, 
               sequence_length * embed_dimension * sizeof(float), 
               cudaMemcpyHostToDevice);
    cudaMemcpy(device_sum_values, sum_values, 
               sequence_length * sizeof(float), 
               cudaMemcpyHostToDevice);
    cudaMemcpy(device_max_values, max_values, 
               sequence_length * sizeof(float), 
               cudaMemcpyHostToDevice);

    // Calculate attention scaling factor
    float attention_scale = 1.0f / sqrt(embed_dimension);

    // Launch configuration
    dim3 block_dim(Block_row_size);  // One thread per row in Query block
    dim3 grid_dim(1);                // Single block for simplicity

    // Launch kernel
    flashAttentionForward<<<grid_dim, block_dim>>>(
        device_Query,
        device_Key,
        device_Value,
        device_Output,
        device_max_values,
        device_sum_values,
        attention_scale
    );

    // Check for kernel launch errors
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Copy results back to host
    cudaMemcpy(Output, device_Output, 
               sequence_length * embed_dimension * sizeof(float), 
               cudaMemcpyDeviceToHost);
    cudaMemcpy(max_values, device_max_values, 
               sequence_length * sizeof(float), 
               cudaMemcpyDeviceToHost);
    cudaMemcpy(sum_values, device_sum_values, 
               sequence_length * sizeof(float), 
               cudaMemcpyDeviceToHost);



// Print Query
std::cout << "Query:" << std::endl;
for (int i = 0; i < sequence_length; i++) {
    for (int j = 0; j < embed_dimension; j++) {
        std::cout << Query[i][j] << " ";
    }
    std::cout << std::endl;
}

// Print Key
std::cout << "Key:" << std::endl;
for (int i = 0; i < sequence_length; i++) {
    for (int j = 0; j < embed_dimension; j++) {
        std::cout << Key[i][j] << " ";
    }
    std::cout << std::endl;
}

// Print Value
std::cout << "Value:" << std::endl;
for (int i = 0; i < sequence_length; i++) {
    for (int j = 0; j < embed_dimension; j++) {
        std::cout << Value[i][j] << " ";
    }
    std::cout << std::endl;
}

// Print Output
std::cout << "Output:" << std::endl;
for (int i = 0; i < sequence_length; i++) {
    for (int j = 0; j < embed_dimension; j++) {
        std::cout << Output[i][j] << " ";
    }
    std::cout << std::endl;
}

Error:
    // Cleanup
    // Device memory
    cudaFree(device_Query);
    cudaFree(device_Key);
    cudaFree(device_Value);
    cudaFree(device_Output);
    cudaFree(device_max_values);
    cudaFree(device_sum_values);

    // Host memory
    delete[] Query;
    delete[] Key;
    delete[] Value;
    delete[] Output;
    delete[] sum_values;
    delete[] max_values;

    return cudaStatus == cudaSuccess ? 0 : 1;
}
