#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// CUDA kernel performing the M-RoPE transformation for one token.
extern "C" __global__ void qwen2vl_mrope_kernel(
    float* q,               // [bs*sl, n_qh * hd]
    float* k,               // [bs*sl, n_kh * hd]
    const float* cos,       // shape: [3, bs*sl, hd]
    const float* sin,       // shape: [3, bs*sl, hd]
    int sl,                 // sequence length
    int bs,                 // batch size
    int n_qh,               // number of Q heads
    int n_kh,               // number of K heads
    int hd,                 // head dimension (must be even)
    int pad_n_qh,           // padded number of Q heads (assumed = n_qh)
    int pad_n_kh,           // padded number of K heads (assumed = n_kh)
    int pad_hd,             // padded head dimension (assumed = hd)
    int mrope_section_t,    // mrope section “t”
    int mrope_section_h,    // mrope section “h”
    bool backward_pass      // if true, perform backward transformation
) {
    // Each thread is responsible for one token.
    int token_id = blockIdx.x * blockDim.x + threadIdx.x;
    int n_row = bs * sl;
    if (token_id >= n_row) return;

    // Each token's Q and K are stored contiguously:
    // Q: [n_qh, hd] and K: [n_kh, hd].
    float* q_token = q + token_id * n_qh * hd;
    float* k_token = k + token_id * n_kh * hd;

    // cos and sin arrays are arranged in three contiguous blocks:
    // Section 0: t_cos/t_sin, Section 1: h_cos/h_sin, Section 2: w_cos/w_sin.
    const int token_offset = token_id * hd;
    const float* t_cos = cos + token_offset;
    const float* h_cos = cos + bs * sl * hd + token_offset;
    const float* w_cos = cos + 2 * bs * sl * hd + token_offset;
    const float* t_sin = sin + token_offset;
    const float* h_sin = sin + bs * sl * hd + token_offset;
    const float* w_sin = sin + 2 * bs * sl * hd + token_offset;

    // For the rotary computation we use only the first half of the head dimension.
    int half_hd = hd / 2;
    int h_end = mrope_section_t + mrope_section_h;  // boundary for second section

    // Process each Q head for this token.
    for (int head = 0; head < n_qh; head++) {
        float* q_head_ptr = q_token + head * hd;
        for (int d = 0; d < half_hd; d++) {
            float q1 = q_head_ptr[d];
            float q2 = q_head_ptr[d + half_hd];

            float cos_val = 0.f, sin_val = 0.f;
            if (d < mrope_section_t) {
                cos_val = t_cos[d];
                sin_val = t_sin[d];
            } else if (d < h_end) {
                cos_val = h_cos[d];
                sin_val = h_sin[d];
            } else if (d < half_hd) {
                cos_val = w_cos[d];
                sin_val = w_sin[d];
            }
            float new_q1, new_q2;
            if (!backward_pass) {
                new_q1 = q1 * cos_val - q2 * sin_val;
                new_q2 = q2 * cos_val + q1 * sin_val;
            } else {
                new_q1 = q1 * cos_val + q2 * sin_val;
                new_q2 = q2 * cos_val - q1 * sin_val;
            }
            q_head_ptr[d] = new_q1;
            q_head_ptr[d + half_hd] = new_q2;
        }
    }

    // Process each K head for this token.
    for (int head = 0; head < n_kh; head++) {
        float* k_head_ptr = k_token + head * hd;
        for (int d = 0; d < half_hd; d++) {
            float k1 = k_head_ptr[d];
            float k2 = k_head_ptr[d + half_hd];

            float cos_val = 0.f, sin_val = 0.f;
            if (d < mrope_section_t) {
                cos_val = t_cos[d];
                sin_val = t_sin[d];
            } else if (d < h_end) {
                cos_val = h_cos[d];
                sin_val = h_sin[d];
            } else if (d < half_hd) {
                cos_val = w_cos[d];
                sin_val = w_sin[d];
            }
            float new_k1, new_k2;
            if (!backward_pass) {
                new_k1 = k1 * cos_val - k2 * sin_val;
                new_k2 = k2 * cos_val + k1 * sin_val;
            } else {
                new_k1 = k1 * cos_val + k2 * sin_val;
                new_k2 = k2 * cos_val - k1 * sin_val;
            }
            k_head_ptr[d] = new_k1;
            k_head_ptr[d + half_hd] = new_k2;
        }
    }
}


void qwen2vl_mrope_forward(
    float* d_q,             // device pointer for q tensor; shape: [bs*sl, n_qh * hd]
    float* d_k,             // device pointer for k tensor; shape: [bs*sl, n_kh * hd]
    const float* d_cos,     // device pointer for cos; shape: [3, bs*sl, hd]
    const float* d_sin,     // device pointer for sin; shape: [3, bs*sl, hd]
    int bs,
    int sl,
    int n_qh,
    int n_kh,
    int hd,
    int mrope_section_t,
    int mrope_section_h
) {
    int pad_n_qh = n_qh;
    int pad_n_kh = n_kh;
    int pad_hd = hd;

    int n_row = bs * sl;
    int threads = 256;
    int blocks = (n_row + threads - 1) / threads;
    qwen2vl_mrope_kernel<<<blocks, threads>>>(d_q, d_k, d_cos, d_sin,
                                               sl, bs, n_qh, n_kh, hd,
                                               pad_n_qh, pad_n_kh, pad_hd,
                                               mrope_section_t, mrope_section_h,
                                               false);
    cudaDeviceSynchronize();
}


void qwen2vl_mrope_backward(
    float* d_dq,            // device pointer for dq tensor; shape: [bs*sl, n_qh * hd]
    float* d_dk,            // device pointer for dk tensor; shape: [bs*sl, n_kh * hd]
    const float* d_cos,     // device pointer for cos; shape: [3, bs*sl, hd]
    const float* d_sin,     // device pointer for sin; shape: [3, bs*sl, hd]
    int bs,
    int sl,
    int n_qh,
    int n_kh,
    int hd,
    int mrope_section_t,
    int mrope_section_h
) {
    int pad_n_qh = n_qh;
    int pad_n_kh = n_kh;
    int pad_hd = hd;

    int n_row = bs * sl;
    int threads = 256;
    int blocks = (n_row + threads - 1) / threads;
    qwen2vl_mrope_kernel<<<blocks, threads>>>(d_dq, d_dk, d_cos, d_sin,
                                               sl, bs, n_qh, n_kh, hd,
                                               pad_n_qh, pad_n_kh, pad_hd,
                                               mrope_section_t, mrope_section_h,
                                               true);
    cudaDeviceSynchronize();
}

int main() {
    // Example dimensions.
    const int bs = 2;               // batch size
    const int sl = 4;               // sequence length
    const int n_qh = 2;             // number of Q heads 
    const int n_kh = 2;             // number of K heads 
    const int hd = 8;               // head dimension (must be even)
    const int mrope_section_t = 3;  // example value (must be <= hd/2)
    const int mrope_section_h = 2;  // example value

    int n_row = bs * sl;
    size_t size_q = n_row * n_qh * hd * sizeof(float);
    size_t size_k = n_row * n_kh * hd * sizeof(float);
    size_t size_cos = 3 * n_row * hd * sizeof(float);
    size_t size_sin = size_cos;

    float* h_q = (float*)malloc(size_q);
    float* h_k = (float*)malloc(size_k);
    float* h_cos = (float*)malloc(size_cos);
    float* h_sin = (float*)malloc(size_sin);

    for (size_t i = 0; i < size_q / sizeof(float); i++) {
        h_q[i] = 1.0f;
    }
    for (size_t i = 0; i < size_k / sizeof(float); i++) {
        h_k[i] = 1.0f;
    }
    for (size_t i = 0; i < size_cos / sizeof(float); i++) {
        h_cos[i] = cosf(i * 0.01f);
        h_sin[i] = sinf(i * 0.01f);
    }


    float *d_q, *d_k, *d_cos, *d_sin;
    cudaMalloc(&d_q, size_q);
    cudaMalloc(&d_k, size_k);
    cudaMalloc(&d_cos, size_cos);
    cudaMalloc(&d_sin, size_sin);


    cudaMemcpy(d_q, h_q, size_q, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k, size_k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cos, h_cos, size_cos, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sin, h_sin, size_sin, cudaMemcpyHostToDevice);

   
    qwen2vl_mrope_forward(d_q, d_k, d_cos, d_sin,
                          bs, sl, n_qh, n_kh, hd,
                          mrope_section_t, mrope_section_h);


    cudaMemcpy(h_q, d_q, size_q, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_k, d_k, size_k, cudaMemcpyDeviceToHost);

   
    printf("Transformed Q values:\n");
    for (int i = 0; i < n_row; i++) {
        printf("Token %d:\n", i);
        for (int head = 0; head < n_qh; head++) {
            printf("  Q head %d: ", head);
            for (int d = 0; d < hd; d++) {
                int index = i * n_qh * hd + head * hd + d;
                printf("%0.3f ", h_q[index]);
            }
            printf("\n");
        }
    }

    printf("\nTransformed K values:\n");
    for (int i = 0; i < n_row; i++) {
        printf("Token %d:\n", i);
        for (int head = 0; head < n_kh; head++) {
            printf("  K head %d: ", head);
            for (int d = 0; d < hd; d++) {
                int index = i * n_kh * hd + head * hd + d;
                printf("%0.3f ", h_k[index]);
            }
            printf("\n");
        }
    }

   
    free(h_q);
    free(h_k);
    free(h_cos);
    free(h_sin);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_cos);
    cudaFree(d_sin);

    printf("\nM-RoPE transformation completed.\n");
    return 0;
}
