#include <hip/hip_runtime.h>
#include <iostream>
#include <fstream>
#include <iomanip>

#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error in %s:%d: %s\n", __FILE__, __LINE__, \
                hipGetErrorString(err)); \
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
    const int threshold = 20;   // Threshold for ELL storage

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
    HIP_CHECK(hipMalloc(&d_global_coo_counter, sizeof(int)));
    HIP_CHECK(hipMemset(d_global_coo_counter, 0, sizeof(int)));  // Initialize to 0

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
    HIP_CHECK(hipMalloc(&d_A, N * M * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_X, M * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_data_ell, N * threshold * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_data_coo, N * M * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_indices_ell, N * threshold * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_row_coo, N * M * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_col_coo, N * M * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_output_matrix, N * sizeof(float)));

    // Copy data to device
    HIP_CHECK(hipMemcpy(d_A, A, N * M * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_X, X, M * sizeof(float), hipMemcpyHostToDevice));

    // Get device properties
    int device;
    HIP_CHECK(hipGetDevice(&device));
    hipDeviceProp_t deviceProp;
    HIP_CHECK(hipGetDeviceProperties(&deviceProp, device));

    // Configure kernel launch parameters
    int block_size = 256;  // Use a reasonable block size
    int num_blocks = (N + block_size - 1) / block_size;

    // Setup timing
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    // Record start time
    HIP_CHECK(hipEventRecord(start));

    // Launch kernel
    hipLaunchKernelGGL(ELL_kernel, dim3(num_blocks), dim3(block_size), 0, 0,
                       d_A, d_X, d_data_ell, d_indices_ell, d_data_coo, d_row_coo,
                       d_col_coo, d_output_matrix, threshold, N, M, d_global_coo_counter);

    // Check for kernel launch errors
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // Record stop time
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float milliseconds = 0;
    HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));
    std::cout << "HIP kernel time: " << milliseconds / 1000.0 << " seconds" << std::endl;

    // Copy results back to host
    HIP_CHECK(hipMemcpy(data_ell, d_data_ell, N * threshold * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(data_coo, d_data_coo, N * M * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(indices_ell, d_indices_ell, N * threshold * sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(row_coo, d_row_coo, N * M * sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(col_coo, d_col_coo, N * M * sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(output_matrix, d_output_matrix, N * sizeof(float), hipMemcpyDeviceToHost));

    // Copy global_coo_counter back to host to verify the number of COO elements
    int h_global_coo_counter;
    HIP_CHECK(hipMemcpy(&h_global_coo_counter, d_global_coo_counter, sizeof(int), hipMemcpyDeviceToHost));

    // Print first 10 COO elements
    for (int i = 0; i < 10; ++i) {
        std::cout << "COO[" << i << "]: val = " << data_coo[i] 
                  << ", row = " << row_coo[i] 
                  << ", col = " << col_coo[i] << std::endl;
    }

    // Write results to file
    std::ofstream output_file("hip_results.txt");
    if (!output_file.is_open()) {
        std::cerr << "Failed to open output file!" << std::endl;
        return EXIT_FAILURE;
    }
    
    for (int i = 0; i < N; i++) {
        output_file << std::fixed << std::setprecision(10) << output_matrix[i] << "\n";
    }
    output_file.close();
    std::cout << "Wrote " << N << " values to hip_results.txt" << std::endl;

    // Clean up events
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    // Free device memory
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_X));
    HIP_CHECK(hipFree(d_data_ell));
    HIP_CHECK(hipFree(d_data_coo));
    HIP_CHECK(hipFree(d_indices_ell));
    HIP_CHECK(hipFree(d_row_coo));
    HIP_CHECK(hipFree(d_col_coo));
    HIP_CHECK(hipFree(d_output_matrix));
    HIP_CHECK(hipFree(d_global_coo_counter));

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