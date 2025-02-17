#include "matrix_mult.h"
#include <random>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <cmath>

void generate_random_sparse_matrix(CSRMatrix& matrix, float density) {
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::uniform_real_distribution<> val_dis(-1.0, 1.0);
    
    // Clear existing data
    matrix.values.clear();
    matrix.col_indices.clear();
    matrix.row_ptr.clear();
    
    // Initialize row_ptr with zeros
    matrix.row_ptr.resize(matrix.rows + 1, 0);
    
    // First pass: count non-zero elements per row
    for (int i = 0; i < matrix.rows; ++i) {
        for (int j = 0; j < matrix.cols; ++j) {
            if (dis(gen) < density) {
                matrix.values.push_back(val_dis(gen));
                matrix.col_indices.push_back(j);
                matrix.row_ptr[i + 1]++;
            }
        }
    }
    
    // Compute cumulative sum for row_ptr
    for (int i = 1; i <= matrix.rows; ++i) {
        matrix.row_ptr[i] += matrix.row_ptr[i - 1];
    }
}

void generate_random_sparse_matrix(COOMatrix& mat, float density) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> val_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> sparsity_dist(0.0f, 1.0f);

    // Calculate number of non-zero elements
    int total_elements = mat.rows * mat.cols;
    mat.nnz = static_cast<int>(density * total_elements);

    // Generate non-zero elements
    mat.values.clear();
    mat.row_indices.clear();
    mat.col_indices.clear();

    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            if (sparsity_dist(gen) < density) {
                mat.values.push_back(val_dist(gen));
                mat.row_indices.push_back(i);
                mat.col_indices.push_back(j);
            }
        }
    }

    // Sort by row and column indices
    std::vector<std::tuple<int, int, float>> elements;
    for (size_t i = 0; i < mat.values.size(); ++i) {
        elements.emplace_back(mat.row_indices[i], mat.col_indices[i], mat.values[i]);
    }
    std::sort(elements.begin(), elements.end());

    // Update arrays
    mat.values.clear();
    mat.row_indices.clear();
    mat.col_indices.clear();
    for (const auto& elem : elements) {
        mat.row_indices.push_back(std::get<0>(elem));
        mat.col_indices.push_back(std::get<1>(elem));
        mat.values.push_back(std::get<2>(elem));
    }
}

void generate_random_sparse_matrix(BlockCSRMatrix& matrix, float density) {
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::uniform_real_distribution<> val_dis(-1.0, 1.0);
    
    // Clear existing data
    matrix.values.clear();
    matrix.col_indices.clear();
    matrix.row_ptr.clear();
    
    // Initialize row_ptr with zeros
    int num_block_rows = matrix.rows / matrix.block_size;
    matrix.row_ptr.resize(num_block_rows + 1, 0);
    
    // First pass: count non-zero blocks per block row
    for (int i = 0; i < num_block_rows; ++i) {
        for (int j = 0; j < matrix.cols / matrix.block_size; ++j) {
            if (dis(gen) < density) {
                // Generate random values for the block
                for (int bi = 0; bi < matrix.block_size; ++bi) {
                    for (int bj = 0; bj < matrix.block_size; ++bj) {
                        matrix.values.push_back(val_dis(gen));
                    }
                }
                matrix.col_indices.push_back(j);
                matrix.row_ptr[i + 1]++;
            }
        }
    }
    
    // Compute cumulative sum for row_ptr
    for (int i = 1; i <= num_block_rows; ++i) {
        matrix.row_ptr[i] += matrix.row_ptr[i - 1];
    }
}

void generate_random_matrix(float* mat, int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = dist(gen);
    }
}

double compare_matrices(const float* A, const float* B, int rows, int cols) {
    double max_diff = 0.0;
    for (int i = 0; i < rows * cols; ++i) {
        double diff = std::abs(A[i] - B[i]);
        max_diff = std::max(max_diff, diff);
    }
    return max_diff;
}

void print_performance_results(const std::vector<PerfResult>& results) {
    // Print header
    std::cout << std::setw(15) << "Format"
              << std::setw(15) << "Time (ms)"
              << std::setw(15) << "GFLOPS"
              << std::setw(15) << "Max Diff" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    // Print results
    for (const auto& result : results) {
        std::cout << std::setw(15) << result.format
                  << std::setw(15) << std::fixed << std::setprecision(3) << result.time_ms
                  << std::setw(15) << std::fixed << std::setprecision(2) << result.gflops
                  << std::setw(15) << std::scientific << std::setprecision(3) << result.max_diff
                  << std::endl;
    }
    std::cout.flush();  // Ensure output is written immediately
} 