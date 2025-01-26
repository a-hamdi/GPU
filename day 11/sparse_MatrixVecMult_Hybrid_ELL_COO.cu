#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void ELL_kernel(const float* A, const float* X, float* data_ell,
                           int* indices_ell, float* data_coo, int* row_coo,
                           int* col_coo, float* output_matrix, const int threshold,
                           const int N, const int M, int* global_coo_counter) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    int counter = 0;

    // Process row
    for (int col = 0; col < M; ++col) {
        float val = A[row * M + col];
        if (val != 0) {
            if (counter < threshold) {
                // ELL format storage
                data_ell[counter * N + row] = val;
                indices_ell[counter * N + row] = col;
                counter++;
            } else {
                // COO format storage
                int coo_index = atomicAdd(global_coo_counter, 1);  // Atomic global counter
                data_coo[coo_index] = val;
                row_coo[coo_index] = row;
                col_coo[coo_index] = col;
            }
        }
    }

    // Fill unused ELL slots with zeros
    for (int i = counter; i < threshold; ++i) {
        data_ell[i * N + row] = 0;
        indices_ell[i * N + row] = -1;
    }

    // Matrix-vector multiplication using ELL format
    float acc = 0.0f;
    for (int p = 0; p < threshold; ++p) {
        int index = indices_ell[p * N + row];
        if (index != -1) {
            acc += data_ell[p * N + row] * X[index];
        }
    }

    // Add COO contribution
    for (int i = 0; i < *global_coo_counter; ++i) {
        if (row_coo[i] == row) {  // Verify this COO element belongs to the current row
            acc += data_coo[i] * X[col_coo[i]];
        }
    }

    output_matrix[row] = acc;

}

int main() {
    const int N = 1000;        // Rows
    const int M = 1000;        // Columns
    const int threshold = 20; // Threshold for ELL storage

    // Host arrays - using dynamic allocation
    float* A = new float[N * M];
    float* data_ell = new float[N * threshold]();  // Initialize to zero
    float* data_coo = new float[N * M]();
    int* indices_ell = new int[N * threshold]();
    int* row_coo = new int[N * M]();
    int* col_coo = new int[N * M]();
    float* X = new float[M];
    float* output_matrix = new float[N];
int* d_global_coo_counter;
CUDA_CHECK(cudaMalloc(&d_global_coo_counter, sizeof(int)));
CUDA_CHECK(cudaMemset(d_global_coo_counter, 0, sizeof(int)));  // Initialize to 0
    // Initialize matrix A and vector X
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            A[i * M + j] = (i + j) % 3 == 0 ? i + j : 0;
        }
    }
    for (int i = 0; i < M; i++) {
        X[i] = 1.0f;
    }

    // Device pointers
    float *d_A, *d_X, *d_data_ell, *d_data_coo, *d_output_matrix;
    int *d_indices_ell, *d_row_coo, *d_col_coo;

    // Allocate device memory with error checking
    CUDA_CHECK(cudaMalloc(&d_A, N * M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_X, M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_data_ell, N * threshold * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_data_coo, N * M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_indices_ell, N * threshold * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_row_coo, N * M * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_coo, N * M * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output_matrix, N * sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, A, N * M * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_X, X, M * sizeof(float), cudaMemcpyHostToDevice));

    // Get device properties
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    // Configure kernel launch parameters
    int block_size = 256;  // Use a reasonable block size
    int num_blocks = (N + block_size - 1) / block_size;

    // Setup timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Record start time
    CUDA_CHECK(cudaEventRecord(start));

    // Launch kernel
    ELL_kernel<<<num_blocks, block_size>>>(d_A, d_X, d_data_ell, d_indices_ell,
                                         d_data_coo, d_row_coo, d_col_coo,
                                         d_output_matrix, threshold, N, M,d_global_coo_counter);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Record stop time
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "CUDA kernel time: " << milliseconds / 1000.0 << " seconds" << std::endl;

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(data_ell, d_data_ell, N * threshold * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data_coo, d_data_coo, N * M * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(indices_ell, d_indices_ell, N * threshold * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(row_coo, d_row_coo, N * M * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(col_coo, d_col_coo, N * M * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(output_matrix, d_output_matrix, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Clean up events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    // Copy global_coo_counter back to host to verify the number of COO elements
int h_global_coo_counter;
CUDA_CHECK(cudaMemcpy(&h_global_coo_counter, d_global_coo_counter, sizeof(int), cudaMemcpyDeviceToHost));
for (int i = 0; i < 10; ++i) {
    std::cout << "COO[" << i << "]: val = " << data_coo[i] << ", row = " << row_coo[i] << ", col = " << col_coo[i] << std::endl;
}

FILE *output_file = fopen("cuda_results.txt", "w");
if (output_file == nullptr) {
    std::cerr << "Failed to open output file!" << std::endl;
    return EXIT_FAILURE;
}
for (int i = 0; i < N; i++) {
    fprintf(output_file, "%.10f\n", output_matrix[i]);
}
fclose(output_file);
std::cout << "Wrote " << N << " values to cuda_results.txt" << std::endl;  // Debugging line



    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_data_ell));
    CUDA_CHECK(cudaFree(d_data_coo));
    CUDA_CHECK(cudaFree(d_indices_ell));
    CUDA_CHECK(cudaFree(d_row_coo));
    CUDA_CHECK(cudaFree(d_col_coo));
    CUDA_CHECK(cudaFree(d_output_matrix));

    // Free host memory
    delete[] A;
    delete[] data_ell;
    delete[] data_coo;
    delete[] indices_ell;
    delete[] row_coo;
    delete[] col_coo;
    delete[] X;
    delete[] output_matrix;

    return 0;
}
