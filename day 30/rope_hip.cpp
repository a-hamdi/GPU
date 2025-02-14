#include <hip/hip_runtime.h>
#include <math.h>
#include <iostream>

#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        std::cerr << "HIP error in " << __FILE__ << ":" << __LINE__ << " - " << hipGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__device__ void apply_rotary_embedding(
    float* q,           // query vectors
    float* k,           // key vectors
    const int head_dim, // dimension of each head
    const int position, // absolute position in sequence
    const float base = 10000.0f
) {
    // Process pairs of elements (real, imaginary)
    for (int i = 0; i < head_dim; i += 2) {
        float freq = 1.0f / powf(base, (float)(i) / head_dim);
        float theta = position * freq;
        
        // Calculate rotation matrix elements
        float cos_theta = cosf(theta);
        float sin_theta = sinf(theta);
        
        // Cache original values
        float q_real = q[i];
        float q_img = q[i + 1];
        float k_real = k[i];
        float k_img = k[i + 1];
        
        // Apply rotation to query
        q[i] = q_real * cos_theta - q_img * sin_theta;
        q[i + 1] = q_real * sin_theta + q_img * cos_theta;
        
        // Apply rotation to key
        k[i] = k_real * cos_theta - k_img * sin_theta;
        k[i + 1] = k_real * sin_theta + k_img * cos_theta;
    }
}

__global__ void rope_kernel(
    float* queries,        // [batch_size, seq_len, num_heads, head_dim]
    float* keys,          // [batch_size, seq_len, num_heads, head_dim]
    const int batch_size,
    const int seq_len,
    const int num_heads,
    const int head_dim
) {
    // Calculate global position
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    
    // Calculate batch, sequence position, and head indices
    int batch_idx = idx / (seq_len * num_heads);
    int seq_idx = (idx / num_heads) % seq_len;
    int head_idx = idx % num_heads;
    
    if (batch_idx >= batch_size) return;
    
    // Calculate base pointer offsets
    int base_idx = batch_idx * (seq_len * num_heads * head_dim) + 
                   seq_idx * (num_heads * head_dim) +
                   head_idx * head_dim;
    
    // Apply rotary embedding to this position
    apply_rotary_embedding(
        &queries[base_idx],
        &keys[base_idx],
        head_dim,
        seq_idx
    );
}

// Helper function to launch the kernel
void apply_rope(
    float* d_queries,
    float* d_keys,
    const int batch_size,
    const int seq_len,
    const int num_heads,
    const int head_dim
) {
    dim3 block_size(256);
    dim3 grid_size((batch_size * seq_len * num_heads + block_size.x - 1) / block_size.x);
    
    hipLaunchKernelGGL(rope_kernel,
        grid_size,
        block_size,
        0, // shared memory size
        0, // stream
        d_queries,
        d_keys,
        batch_size,
        seq_len,
        num_heads,
        head_dim
    );
    
    // Check for kernel launch errors
    HIP_CHECK(hipGetLastError());
    // Wait for kernel to finish and check for errors
    HIP_CHECK(hipDeviceSynchronize());
}

// Example usage and test function
void test_rope() {
    const int batch_size = 2;
    const int seq_len = 32;
    const int num_heads = 8;
    const int head_dim = 64;
    
    size_t total_size = batch_size * seq_len * num_heads * head_dim * sizeof(float);
    
    // Allocate host memory
    float* h_queries = new float[total_size / sizeof(float)];
    float* h_keys = new float[total_size / sizeof(float)];
    
    // Initialize with some test data
    for (size_t i = 0; i < total_size / sizeof(float); i++) {
        h_queries[i] = static_cast<float>(i) / 1000.0f;
        h_keys[i] = static_cast<float>(i) / 1000.0f;
    }
    
    // Allocate device memory
    float *d_queries, *d_keys;
    HIP_CHECK(hipMalloc(&d_queries, total_size));
    HIP_CHECK(hipMalloc(&d_keys, total_size));
    
    // Copy data to device
    HIP_CHECK(hipMemcpy(d_queries, h_queries, total_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_keys, h_keys, total_size, hipMemcpyHostToDevice));
    
    // Apply RoPE
    apply_rope(d_queries, d_keys, batch_size, seq_len, num_heads, head_dim);
    
    // Copy results back
    HIP_CHECK(hipMemcpy(h_queries, d_queries, total_size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_keys, d_keys, total_size, hipMemcpyDeviceToHost));
    
    // Print some results
    std::cout << "First few values after RoPE application:\n";
    for (int i = 0; i < 10; i++) {
        std::cout << "Q[" << i << "]: " << h_queries[i] << ", K[" << i << "]: " << h_keys[i] << "\n";
    }
    
    // Cleanup
    delete[] h_queries;
    delete[] h_keys;
    HIP_CHECK(hipFree(d_queries));
    HIP_CHECK(hipFree(d_keys));
}

int main() {
    test_rope();
    return 0;
} 