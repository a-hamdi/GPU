#include <hip/hip_runtime.h>
#include <iostream>
#include <random>
#include <cmath>
#include <fstream>

__global__
void forward_kernel(const float* query_matrix_device_pointer, const float* key_matrix_device_pointer, const float* value_matrix_device_pointer, const int sequence_length, const int embedding_dimension,
                    const int total_columns_in_blocks, const int total_rows_in_blocks, const int block_size_columns, const int block_size_rows, const float softmax_scale,
                    float* sum_matrix_device_pointer, float *max_matrix_device_pointer, float* output_matrix_device_pointer) {
    int thread_index_x = threadIdx.x;
    int block_index_x = blockIdx.x; 
    int block_index_y = blockIdx.y;  // batch and head index

    // Offset into matrices - different for each batch and head
    int qkv_offset = (block_index_x * gridDim.y * sequence_length * embedding_dimension) + (block_index_y * sequence_length * embedding_dimension);
    int lm_offset = (block_index_x * gridDim.y * sequence_length) + (block_index_y * sequence_length);

    // Define shared memory for Q,K,V,S
    HIP_DYNAMIC_SHARED(float, shared_memory)
    int tile_size = block_size_columns * embedding_dimension;
    float* query_matrix_tile = shared_memory;
    float* key_matrix_tile = &shared_memory[tile_size];
    float* value_matrix_tile = &shared_memory[tile_size * 2];
    float* score_matrix_tile = &shared_memory[tile_size * 3];
    float eps = 1e-10;

    for (int column_block_index = 0; column_block_index < total_columns_in_blocks; column_block_index++) {
        // Load key_matrix_tile, value_matrix_tile to shared memory
        for (int embedding_index = 0; embedding_index < embedding_dimension; embedding_index++) {
            key_matrix_tile[(thread_index_x * embedding_dimension) + embedding_index] = key_matrix_device_pointer[qkv_offset + (tile_size * column_block_index) + (thread_index_x * embedding_dimension) + embedding_index];
            value_matrix_tile[(thread_index_x * embedding_dimension) + embedding_index] = value_matrix_device_pointer[qkv_offset + (tile_size * column_block_index) + (thread_index_x * embedding_dimension) + embedding_index];
        }
        __syncthreads();

        for (int row_block_index = 0; row_block_index < total_rows_in_blocks; row_block_index++) {
            // Load query_matrix_tile to shared memory
            for (int embedding_index = 0; embedding_index < embedding_dimension; embedding_index++) {
                query_matrix_tile[(thread_index_x * embedding_dimension) + embedding_index] = query_matrix_device_pointer[qkv_offset + (tile_size * row_block_index) + (thread_index_x * embedding_dimension) + embedding_index];
            }
            float row_max_previous = max_matrix_device_pointer[lm_offset + (block_size_rows * row_block_index) + thread_index_x];
            float row_sum_previous = sum_matrix_device_pointer[lm_offset + (block_size_rows * row_block_index) + thread_index_x];

            // Calculate scores and find max
            float row_max = -INFINITY;
            for (int column_index_inner = 0; column_index_inner < block_size_columns; column_index_inner++) {
                float sum = 0;
                for (int embedding_index = 0; embedding_index < embedding_dimension; embedding_index++) {
                    sum += query_matrix_tile[(thread_index_x * embedding_dimension) + embedding_index] * key_matrix_tile[(column_index_inner * embedding_dimension) + embedding_index];
                }
                sum *= softmax_scale;
                score_matrix_tile[(block_size_columns * thread_index_x) + column_index_inner] = sum;

                if (sum > row_max)
                    row_max = sum;
            }

            // Calculate softmax probabilities
            float row_sum = 0;
            for (int column_index_inner = 0; column_index_inner < block_size_columns; column_index_inner++) {
                score_matrix_tile[(block_size_columns * thread_index_x) + column_index_inner] = expf(score_matrix_tile[(block_size_columns * thread_index_x) + column_index_inner] - row_max);
                row_sum += score_matrix_tile[(block_size_columns * thread_index_x) + column_index_inner];
            }

            float row_max_new = fmax(row_max_previous, row_max);
            float row_sum_new = (expf(row_max_previous - row_max_new) * row_sum_previous) + (expf(row_max - row_max_new) * row_sum);

            // Write output to global memory
            for (int embedding_index = 0; embedding_index < embedding_dimension; embedding_index++) {
                float probability_times_value = 0;
                for (int column_index_inner = 0; column_index_inner < block_size_columns; column_index_inner++) {
                    probability_times_value += score_matrix_tile[(block_size_columns * thread_index_x) + column_index_inner] * value_matrix_tile[(column_index_inner * embedding_dimension) + embedding_index] + eps;
                }
                output_matrix_device_pointer[qkv_offset + (tile_size * row_block_index) + (thread_index_x * embedding_dimension) + embedding_index] = (1 / (eps + row_sum_new)) \
                    * ((row_sum_previous * expf(row_max_previous - row_max_new) * output_matrix_device_pointer[qkv_offset + (tile_size * row_block_index) + (thread_index_x * embedding_dimension) + embedding_index]) \
                    + (expf(row_max - row_max_new + eps) * probability_times_value));
            }
            max_matrix_device_pointer[lm_offset + (block_size_rows * row_block_index) + thread_index_x] = row_max_new;
            sum_matrix_device_pointer[lm_offset + (block_size_rows * row_block_index) + thread_index_x] = row_sum_new;
        }
        __syncthreads();
    }
}

template <typename T>
T* allocateAndInitializeDeviceMemory(size_t size, bool initializeToZero = false, bool initializeToNegativeInfinity = false) {
    T* device_ptr;
    hipMalloc(&device_ptr, size);

    if (initializeToZero) {
        hipMemset(device_ptr, 0, size);
    } else if (initializeToNegativeInfinity) {
        float negative_infinity_host = -INFINITY;
        hipMemset(device_ptr, *reinterpret_cast<int*>(&negative_infinity_host), size);
    } else {
       
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        
        size_t num_elements = size / sizeof(T);
        T* host_data = new T[num_elements];
        for (size_t i = 0; i < num_elements; ++i) {
            host_data[i] = static_cast<T>(dis(gen));
        }
        
        hipMemcpy(device_ptr, host_data, size, hipMemcpyHostToDevice);
        delete[] host_data;
    }

    return device_ptr;
}

template <typename T>
void writeMatrixToFile(T* matrix, const std::string& filename, int batch_size, int num_heads, int sequence_length, int embedding_dimension) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Could not open the file!" << std::endl;
        return;
    }

    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            for (int i = 0; i < sequence_length; ++i) {
                for (int j = 0; j < embedding_dimension; ++j) {
                    file << matrix[(b * num_heads * sequence_length * embedding_dimension) +
                                   (h * sequence_length * embedding_dimension) +
                                   (i * embedding_dimension) + j];
                    if (j < embedding_dimension - 1) {
                        file << ", ";
                    }
                }
                file << std::endl;
            }
            file << std::endl;
        }
    }
    file.close();
}

template <typename T>
void printMatrix(T* matrix, int batch_size, int num_heads, int sequence_length, int embedding_dimension, int rowsToPrint, int colsToPrint) {
    T* host_matrix = new T[batch_size * num_heads * sequence_length * embedding_dimension];
    hipMemcpy(host_matrix, matrix, batch_size * num_heads * sequence_length * embedding_dimension * sizeof(T), hipMemcpyDeviceToHost);

    std::cout << "Matrix:\n";
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            for (int i = 0; i < rowsToPrint; ++i) {
                for (int j = 0; j < colsToPrint; ++j) {
                    std::cout << host_matrix[(b * num_heads * sequence_length * embedding_dimension) +
                                            (h * sequence_length * embedding_dimension) +
                                            (i * embedding_dimension) + j] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }
    delete[] host_matrix;
}

int main() {
    const int batch_size = 1;
    const int num_heads = 1;
    const int sequence_length = 64;
    const int embedding_dimension = 64;

    const int block_size_columns = 32;
    const int block_size_rows = 32;

    const int total_columns_in_blocks = ceil((float)sequence_length / block_size_columns);
    const int total_rows_in_blocks = ceil((float)sequence_length / block_size_rows);
    const float softmax_scale = 1.0f / sqrtf(embedding_dimension);

    size_t matrix_size = batch_size * num_heads * sequence_length * embedding_dimension * sizeof(float);
    size_t vector_size = batch_size * num_heads * sequence_length * sizeof(float);

    // Device memory allocation and initialization
    float* query_matrix_device = allocateAndInitializeDeviceMemory<float>(matrix_size);
    float* key_matrix_device = allocateAndInitializeDeviceMemory<float>(matrix_size);
    float* value_matrix_device = allocateAndInitializeDeviceMemory<float>(matrix_size);
    float* output_matrix_device = allocateAndInitializeDeviceMemory<float>(matrix_size, true);
    float* sum_matrix_device = allocateAndInitializeDeviceMemory<float>(vector_size, false, false);
    float* max_matrix_device = allocateAndInitializeDeviceMemory<float>(vector_size, false, true);
    hipMemset(sum_matrix_device, 0, vector_size);

    // Shared memory size calculation
    const int shared_memory_size = (4 * block_size_columns * embedding_dimension * sizeof(float)) +
                                    (block_size_columns * block_size_rows * sizeof(float));
    int max_shared_memory_size;
    hipDeviceGetAttribute(&max_shared_memory_size, hipDeviceAttributeMaxSharedMemoryPerBlock, 0);

    // Kernel launch configuration
    dim3 grid_dim(batch_size, num_heads);
    dim3 block_dim(block_size_columns);

    hipLaunchKernelGGL(forward_kernel, grid_dim, block_dim, shared_memory_size, 0,
        query_matrix_device, key_matrix_device, value_matrix_device, sequence_length,
        embedding_dimension, total_columns_in_blocks, total_rows_in_blocks, block_size_columns,
        block_size_rows, softmax_scale, sum_matrix_device, max_matrix_device, output_matrix_device);

    hipDeviceSynchronize();

    int rowsToPrint = sequence_length;
    int colsToPrint = embedding_dimension;

    float* query_matrix_host = new float[batch_size * num_heads * sequence_length * embedding_dimension];
    float* key_matrix_host = new float[batch_size * num_heads * sequence_length * embedding_dimension];
    float* value_matrix_host = new float[batch_size * num_heads * sequence_length * embedding_dimension];
    float* output_matrix_host = new float[batch_size * num_heads * sequence_length * embedding_dimension];

    // Copy matrices from device to host
    hipMemcpy(query_matrix_host, query_matrix_device, matrix_size, hipMemcpyDeviceToHost);
    hipMemcpy(key_matrix_host, key_matrix_device, matrix_size, hipMemcpyDeviceToHost);
    hipMemcpy(value_matrix_host, value_matrix_device, matrix_size, hipMemcpyDeviceToHost);
    hipMemcpy(output_matrix_host, output_matrix_device, matrix_size, hipMemcpyDeviceToHost);

    // Write matrices to files
    writeMatrixToFile(query_matrix_host, "query_output.csv", batch_size, num_heads, sequence_length, embedding_dimension);
    writeMatrixToFile(key_matrix_host, "key_output.csv", batch_size, num_heads, sequence_length, embedding_dimension);
    writeMatrixToFile(value_matrix_host, "value_output.csv", batch_size, num_heads, sequence_length, embedding_dimension);
    writeMatrixToFile(output_matrix_host, "output_output.csv", batch_size, num_heads, sequence_length, embedding_dimension);

    // Print matrices
    std::cout << "Q:\n";
    printMatrix(query_matrix_device, batch_size, num_heads, sequence_length, embedding_dimension, rowsToPrint, colsToPrint);
    std::cout << "K:\n";
    printMatrix(key_matrix_device, batch_size, num_heads, sequence_length, embedding_dimension, rowsToPrint, colsToPrint);
    std::cout << "V:\n";
    printMatrix(value_matrix_device, batch_size, num_heads, sequence_length, embedding_dimension, rowsToPrint, colsToPrint);
    std::cout << "O:\n";
    printMatrix(output_matrix_device, batch_size, num_heads, sequence_length, embedding_dimension, rowsToPrint, colsToPrint);

    // Free device memory
    hipFree(query_matrix_device);
    hipFree(key_matrix_device);
    hipFree(value_matrix_device);
    hipFree(output_matrix_device);
    hipFree(sum_matrix_device);
    hipFree(max_matrix_device);

    // Free host memory
    delete[] query_matrix_host;
    delete[] key_matrix_host;
    delete[] value_matrix_host;
    delete[] output_matrix_host;

    return 0;
} 