#include <cuda_runtime.h>
#include <math.h>

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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
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
    
    rope_kernel<<<grid_size, block_size>>>(
        d_queries,
        d_keys,
        batch_size,
        seq_len,
        num_heads,
        head_dim
    );
} 