#include "matrix_mult.h"
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <rocblas/rocblas.h>
#include <rocsparse/rocsparse.h>

// Define HIP_CHECK macro for error handling
#define HIP_CHECK(call) do {                                                        \
    hipError_t err = call;                                                         \
    if (err != hipSuccess) {                                                       \
        printf("HIP error %s:%d: '%s'\n", __FILE__, __LINE__,                     \
               hipGetErrorString(err));                                            \
        exit(1);                                                                   \
    }                                                                              \
} while(0)

// Function to run rocBLAS SGEMM for reference
PerfResult rocblas_sgemm_ref(const float* A, const float* B, float* C, int N, int K, int M) {
    PerfResult result = {0.0, 0.0, "rocBLAS", 0.0};  // Initialize all fields

    // Initialize rocBLAS
    rocblas_handle handle;
    rocblas_create_handle(&handle);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, N * K * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_B, K * M * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_C, N * M * sizeof(float)));

    // Copy data to device
    HIP_CHECK(hipMemcpy(d_A, A, N * K * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, B, K * M * sizeof(float), hipMemcpyHostToDevice));

    // Set up timing
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    HIP_CHECK(hipEventRecord(start));
    rocblas_sgemm(handle,
                  rocblas_operation_none, rocblas_operation_none,
                  M, N, K,
                  &alpha,
                  d_B, M,
                  d_A, K,
                  &beta,
                  d_C, M);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float milliseconds = 0;
    HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));
    result.time_ms = milliseconds;

    // Copy result back to host
    HIP_CHECK(hipMemcpy(C, d_C, N * M * sizeof(float), hipMemcpyDeviceToHost));

    // Calculate GFLOPS
    double operations = 2.0 * N * M * K;
    result.gflops = (operations / (milliseconds * 1e-3)) / 1e9;

    // Cleanup
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    rocblas_destroy_handle(handle);

    return result;
}

void benchmark_dense_algorithms(int N) {
    std::cout << "\nBenchmarking Dense Matrix Multiplication Algorithms (N=" << N << ")\n";
    std::cout << std::string(60, '=') << std::endl;

    // Allocate and initialize matrices
    std::vector<float> A(N * N);
    std::vector<float> B(N * N);
    std::vector<float> C_ref(N * N);
    std::vector<float> C_test(N * N);

    generate_random_matrix(A.data(), N, N);
    generate_random_matrix(B.data(), N, N);

    std::vector<PerfResult> results;

    // Run rocBLAS reference
    auto rocblas_result = rocblas_sgemm_ref(A.data(), B.data(), C_ref.data(), N, N, N);
    results.push_back(rocblas_result);

    // Run Winograd (it's faster and more stable than Strassen)
    auto winograd_result = winograd_multiply(A.data(), B.data(), C_test.data(), N);
    winograd_result.max_diff = compare_matrices(C_ref.data(), C_test.data(), N, N);
    results.push_back(winograd_result);

    print_performance_results(results);
}

void benchmark_sparse_algorithms(int N, float density) {
    std::cout << "\nBenchmarking Sparse Matrix Multiplication Algorithms (N=" << N << ", density=" << density << ")\n";
    std::cout << std::string(60, '=') << std::endl;

    // Create sparse matrices in different formats
    CSRMatrix csr_mat{N, N};
    COOMatrix coo_mat{N, N};
    BlockCSRMatrix bcsr_mat{N, N, 32};  // Using 32x32 blocks

    generate_random_sparse_matrix(csr_mat, density);
    generate_random_sparse_matrix(coo_mat, density);
    generate_random_sparse_matrix(bcsr_mat, density);

    // Dense matrix B and result matrices
    std::vector<float> B(N * N);
    std::vector<float> C_ref(N * N);
    std::vector<float> C_csr(N * N);
    std::vector<float> C_coo(N * N);
    std::vector<float> C_bcsr(N * N);
    std::vector<float> C_rocsparse(N * N);

    generate_random_matrix(B.data(), N, N);

    std::vector<PerfResult> results;

    // Run rocBLAS reference with dense matrices
    std::vector<float> A_dense(N * N, 0.0f);
    for (size_t i = 0; i < csr_mat.values.size(); ++i) {
        int row = std::lower_bound(csr_mat.row_ptr.begin(), csr_mat.row_ptr.end(), i) - csr_mat.row_ptr.begin() - 1;
        A_dense[row * N + csr_mat.col_indices[i]] = csr_mat.values[i];
    }
    auto rocblas_result = rocblas_sgemm_ref(A_dense.data(), B.data(), C_ref.data(), N, N, N);
    results.push_back(rocblas_result);

    // Run rocSPARSE CSR SpMM
    rocsparse_handle handle;
    rocsparse_create_handle(&handle);

    // Allocate device memory
    float *d_values, *d_B, *d_C;
    int *d_row_ptr, *d_col_indices;
    HIP_CHECK(hipMalloc(&d_values, csr_mat.values.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_row_ptr, csr_mat.row_ptr.size() * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_col_indices, csr_mat.col_indices.size() * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_B, N * N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_C, N * N * sizeof(float)));

    // Copy data to device
    HIP_CHECK(hipMemcpy(d_values, csr_mat.values.data(), csr_mat.values.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_row_ptr, csr_mat.row_ptr.data(), csr_mat.row_ptr.size() * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_col_indices, csr_mat.col_indices.data(), csr_mat.col_indices.size() * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, B.data(), N * N * sizeof(float), hipMemcpyHostToDevice));

    // Create matrix descriptors
    rocsparse_mat_descr descr;
    rocsparse_create_mat_descr(&descr);
    rocsparse_set_mat_type(descr, rocsparse_matrix_type_general);
    rocsparse_set_mat_index_base(descr, rocsparse_index_base_zero);

    // Set up timing
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    HIP_CHECK(hipEventRecord(start));
    rocsparse_scsrmm(handle, rocsparse_operation_none, rocsparse_operation_none,
                     N, N, N, csr_mat.values.size(),
                     &alpha, descr, d_values, d_row_ptr, d_col_indices,
                     d_B, N, &beta, d_C, N);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float milliseconds = 0;
    HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));

    // Copy result back to host
    HIP_CHECK(hipMemcpy(C_rocsparse.data(), d_C, N * N * sizeof(float), hipMemcpyDeviceToHost));

    // Calculate GFLOPS for rocSPARSE
    double operations = 2.0 * csr_mat.values.size() * N;
    double gflops = (operations / (milliseconds * 1e-3)) / 1e9;

    PerfResult rocsparse_result = {milliseconds, gflops, "rocSPARSE", 
                                  compare_matrices(C_ref.data(), C_rocsparse.data(), N, N)};
    results.push_back(rocsparse_result);

    // Run CSR SpMM
    auto csr_result = sparse_gemm_csr(csr_mat, B.data(), C_csr.data(), N, N, N);
    csr_result.max_diff = compare_matrices(C_ref.data(), C_csr.data(), N, N);
    results.push_back(csr_result);

    // Run COO SpMM
    auto coo_result = sparse_gemm_coo(coo_mat, B.data(), C_coo.data(), N, N, N);
    coo_result.max_diff = compare_matrices(C_ref.data(), C_coo.data(), N, N);
    results.push_back(coo_result);

    // Run Block-CSR SpMM
    auto bcsr_result = sparse_gemm_block_csr(bcsr_mat, B.data(), C_bcsr.data(), N, N, N);
    bcsr_result.max_diff = compare_matrices(C_ref.data(), C_bcsr.data(), N, N);
    results.push_back(bcsr_result);

    print_performance_results(results);

    // Cleanup rocSPARSE resources
    rocsparse_destroy_mat_descr(descr);
    rocsparse_destroy_handle(handle);
    HIP_CHECK(hipFree(d_values));
    HIP_CHECK(hipFree(d_row_ptr));
    HIP_CHECK(hipFree(d_col_indices));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
}

int main() {
    // Test matrix sizes
    std::vector<int> sizes = {256, 512};  // Testing with smaller sizes
    std::vector<float> densities = {0.1f};  // Just one density value

    // Benchmark dense algorithms
    for (int N : sizes) {
        benchmark_dense_algorithms(N);
    }

    // Benchmark sparse algorithms
    for (int N : sizes) {
        for (float density : densities) {
            benchmark_sparse_algorithms(N, density);
        }
    }

    return 0;
} 