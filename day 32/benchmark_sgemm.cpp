#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include "sgemm.h"
#include "kernel3_registers.h"

// Helper function to initialize matrices with random values
void initialize_matrix(std::vector<float>& matrix, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (int i = 0; i < size * size; i++) {
        matrix[i] = dis(gen);
    }
}

// Helper function to calculate difference between matrices
float matrix_diff(const std::vector<float>& a, const std::vector<float>& b, int size) {
    float max_diff = 0.0f;
    for (int i = 0; i < size * size; i++) {
        float diff = std::abs(a[i] - b[i]);
        max_diff = std::max(max_diff, diff);
    }
    return max_diff;
}

// Helper function to calculate GFLOPS
double calculate_gflops(int N, double time_ms) {
    // For matrix multiplication: 2 * N^3 operations
    double operations = 2.0 * std::pow(N, 3);
    double time_s = time_ms / 1000.0;
    return (operations / time_s) / 1e9;
}

// Helper function to check HIP errors
#define CHECK_HIP(cmd) \
    do { \
        hipError_t error = cmd; \
        if (error != hipSuccess) { \
            std::cerr << "HIP error: " << hipGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

int main() {
    std::cout << "Starting SGEMM benchmark..." << std::endl;

    // Test different matrix sizes
    std::vector<int> matrix_sizes = {1024, 2048, 4096, 8192};
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Initialize rocBLAS
    rocblas_handle handle;
    rocblas_status status = rocblas_create_handle(&handle);
    if (status != rocblas_status_success) {
        std::cerr << "Failed to initialize rocBLAS" << std::endl;
        return 1;
    }

    std::cout << std::setw(10) << "Size" 
              << std::setw(20) << "Custom (GFLOPS)"
              << std::setw(20) << "rocBLAS (GFLOPS)"
              << std::setw(15) << "Max Diff"
              << std::setw(15) << "Time (ms)" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    for (int N : matrix_sizes) {
        std::cout << "Testing size " << N << "x" << N << "..." << std::endl;

        // Allocate host memory
        std::vector<float> h_a(N * N);
        std::vector<float> h_b(N * N);
        std::vector<float> h_c_custom(N * N, 0.0f);
        std::vector<float> h_c_rocblas(N * N, 0.0f);

        std::cout << "Initializing matrices..." << std::endl;
        // Initialize matrices
        initialize_matrix(h_a, N);
        initialize_matrix(h_b, N);

        // Allocate device memory
        float *d_a, *d_b, *d_c_custom, *d_c_rocblas;
        std::cout << "Allocating device memory..." << std::endl;
        CHECK_HIP(hipMalloc(&d_a, N * N * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_b, N * N * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_c_custom, N * N * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_c_rocblas, N * N * sizeof(float)));

        // Copy data to device
        std::cout << "Copying data to device..." << std::endl;
        CHECK_HIP(hipMemcpy(d_a, h_a.data(), N * N * sizeof(float), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_b, h_b.data(), N * N * sizeof(float), hipMemcpyHostToDevice));

        // Benchmark custom kernel
        std::cout << "Running custom kernel..." << std::endl;
        Kernel3Registers kernel;
        kernel.init();

        auto start = std::chrono::high_resolution_clock::now();
        kernel.run(d_a, d_b, d_c_custom, alpha, beta, N);
        CHECK_HIP(hipDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        auto custom_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double custom_time_ms = custom_duration.count() / 1000.0;
        double custom_gflops = calculate_gflops(N, custom_time_ms);

        // Benchmark rocBLAS
        std::cout << "Running rocBLAS..." << std::endl;
        start = std::chrono::high_resolution_clock::now();
        status = rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none,
                      N, N, N,
                      &alpha,
                      d_a, N,
                      d_b, N,
                      &beta,
                      d_c_rocblas, N);
        if (status != rocblas_status_success) {
            std::cerr << "rocBLAS SGEMM failed" << std::endl;
            return 1;
        }
        CHECK_HIP(hipDeviceSynchronize());
        end = std::chrono::high_resolution_clock::now();
        auto rocblas_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double rocblas_time_ms = rocblas_duration.count() / 1000.0;
        double rocblas_gflops = calculate_gflops(N, rocblas_time_ms);

        // Copy results back to host
        std::cout << "Copying results back to host..." << std::endl;
        CHECK_HIP(hipMemcpy(h_c_custom.data(), d_c_custom, N * N * sizeof(float), hipMemcpyDeviceToHost));
        CHECK_HIP(hipMemcpy(h_c_rocblas.data(), d_c_rocblas, N * N * sizeof(float), hipMemcpyDeviceToHost));

        // Calculate difference
        float max_diff = matrix_diff(h_c_custom, h_c_rocblas, N);

        // Print results
        std::cout << std::setw(10) << N
                  << std::setw(20) << std::fixed << std::setprecision(2) << custom_gflops
                  << std::setw(20) << rocblas_gflops
                  << std::setw(15) << std::scientific << std::setprecision(3) << max_diff
                  << std::setw(15) << std::fixed << std::setprecision(2) << custom_time_ms << std::endl;

        // Cleanup
        CHECK_HIP(hipFree(d_a));
        CHECK_HIP(hipFree(d_b));
        CHECK_HIP(hipFree(d_c_custom));
        CHECK_HIP(hipFree(d_c_rocblas));
    }

    rocblas_destroy_handle(handle);
    return 0;
} 