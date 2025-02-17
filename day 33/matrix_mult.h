#pragma once

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <vector>
#include <string>

// Sparse matrix formats
enum class SparseFormat {
    CSR,
    COO,
    BLOCK_CSR
};

// Structure for sparse matrix in CSR format
struct CSRMatrix {
    std::vector<float> values;     // Non-zero values
    std::vector<int> row_ptr;      // Row pointers
    std::vector<int> col_indices;  // Column indices
    int rows;
    int cols;
    int nnz;  // Number of non-zero elements

    CSRMatrix(int r, int c) : rows(r), cols(c), nnz(0) {
        row_ptr.resize(r + 1, 0);
    }
};

// Structure for sparse matrix in COO format
struct COOMatrix {
    std::vector<float> values;     // Non-zero values
    std::vector<int> row_indices;  // Row indices
    std::vector<int> col_indices;  // Column indices
    int rows;
    int cols;
    int nnz;  // Number of non-zero elements

    COOMatrix(int r, int c) : rows(r), cols(c), nnz(0) {}
};

// Structure for sparse matrix in Block-CSR format
struct BlockCSRMatrix {
    std::vector<float> values;     // Non-zero blocks
    std::vector<int> row_ptr;      // Row pointers
    std::vector<int> col_indices;  // Column indices
    int rows;
    int cols;
    int block_size;                // Size of each block

    BlockCSRMatrix(int r, int c, int bs) : rows(r), cols(c), block_size(bs) {
        int block_rows = (r + bs - 1) / bs;
        row_ptr.resize(block_rows + 1, 0);
    }
};

// Performance result structure
struct PerfResult {
    double time_ms;        // Execution time in milliseconds
    double gflops;         // Performance in GFLOPS
    std::string format;    // Format or algorithm used
    double max_diff;       // Maximum difference from reference
};

// Function declarations for sparse GEMM
PerfResult sparse_gemm_csr(const CSRMatrix& A, const float* B, float* C, int N, int K, int M);
PerfResult sparse_gemm_coo(const COOMatrix& A, const float* B, float* C, int N, int K, int M);
PerfResult sparse_gemm_block_csr(const BlockCSRMatrix& A, const float* B, float* C, int N, int K, int M);

// Function declarations for Strassen algorithm
PerfResult strassen_multiply(const float* A, const float* B, float* C, int N);

// Function declarations for Winograd algorithm
PerfResult winograd_multiply(const float* A, const float* B, float* C, int N);

// Function declarations for rocBLAS reference implementation
PerfResult rocblas_sgemm_ref(const float* A, const float* B, float* C, int N, int K, int M);

// Utility functions
void generate_random_sparse_matrix(CSRMatrix& mat, float density);
void generate_random_sparse_matrix(COOMatrix& mat, float density);
void generate_random_sparse_matrix(BlockCSRMatrix& mat, float density);
void generate_random_matrix(float* mat, int rows, int cols);
double compare_matrices(const float* A, const float* B, int rows, int cols);
void print_performance_results(const std::vector<PerfResult>& results); 