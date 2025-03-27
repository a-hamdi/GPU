#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <algorithm>

#define MAX_FUSED_SIZE 65536

int next_power_of_2(int n) {
    int p = 1;
    while (p < n) p *= 2;
    return p;
}

__global__ void jsd_kernel(
    const float* __restrict__ X, int X_stride,
    const float* __restrict__ Y, int Y_stride,
    float* __restrict__ loss, int loss_stride,
    float* __restrict__ dX, int dX_stride,
    const int* __restrict__ labels,
    float beta,
    int n_non_ignore,
    int ignore_index,
    int n_cols,
    bool has_label)
{
    int row = blockIdx.x;
    const float* X_row = X + row * X_stride;
    const float* Y_row = Y + row * Y_stride;
    float* loss_row = loss + row * loss_stride;
    float* dX_row = dX + row * dX_stride;

    if (has_label) {
        int label = labels[row];
        if (label == ignore_index) {
            for (int col = threadIdx.x; col < n_cols; col += blockDim.x) {
                dX_row[col] = 0.0f;
            }
            return;
        }
    }

    for (int i = 0; i < n_cols; i += blockDim.x) {
        int idx = i + threadIdx.x;
        float x_val = (idx < n_cols) ? X_row[idx] : -INFINITY;
        float y_val = (idx < n_cols) ? Y_row[idx] : -INFINITY;

        float max_val = -INFINITY;
        extern __shared__ float sdata[];
        if (beta == 0.0f) {
            sdata[threadIdx.x] = y_val;
        } else if (beta == 1.0f) {
            sdata[threadIdx.x] = x_val;
        } else {
            sdata[threadIdx.x] = fmaxf(x_val, y_val);
        }
        __syncthreads();

        for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
            if (threadIdx.x < offset) {
                sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + offset]);
            }
            __syncthreads();
        }
        max_val = sdata[0];

        float l = 0.0f;
        float dx = 0.0f;
        if (beta == 0.0f) {
            float y_shifted = y_val - max_val;
            float y_prob = expf(y_shifted) * expf(max_val); 
            l = y_prob * (y_val - x_val);
            dx = -y_prob;
        }
        else if (beta == 1.0f) {
            float x_shifted = x_val - max_val;
            float x_prob = expf(x_shifted) * expf(max_val); 
            l = x_prob * (x_val - y_val);
            dx = l + x_prob;
        }
        else {
            float x_shifted = x_val - max_val;
            float y_shifted = y_val - max_val;
            float exp_max = expf(max_val);
            float Q = expf(x_shifted) * exp_max; 
            float P = expf(y_shifted) * exp_max; 
            float beta_P = beta * P;
            float one_minus_beta_Q = (1.0f - beta) * Q;
            float M = beta_P + one_minus_beta_Q;
            float log_M = logf(M);
            l = beta_P * y_val + one_minus_beta_Q * x_val - M * log_M;
            dx = one_minus_beta_Q * (x_val - log_M);
        }
        float scale = 1.0f / n_non_ignore;
        l *= scale;
        dx *= scale;

        if (idx < n_cols) {
            loss_row[idx] = l;
            dX_row[idx] = dx;
        }
        __syncthreads();  
    }
}

void jsd_forward(
    const float* d_input,
    const float* d_target,
    const int* d_shift_labels,  
    float beta,
    int ignore_index,
    bool has_label,
    int BT,
    int V,
    float* d_loss,
    float* d_dX,
    int n_non_ignore)
{
    int blockSize = next_power_of_2(V);
    if (blockSize > MAX_FUSED_SIZE) {
        blockSize = MAX_FUSED_SIZE;
    }
    size_t shmem_bytes = blockSize * sizeof(float);
    dim3 grid(BT);
    dim3 block(blockSize);
    jsd_kernel<<<grid, block, shmem_bytes>>>(
        d_input, V,      
        d_target, V,
        d_loss, V,
        d_dX, V,
        d_shift_labels,
        beta,
        n_non_ignore,
        ignore_index,
        V,
        has_label);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("jsd_kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

__global__ void jsd_backward_kernel(const float* dX_in, float* dX_out, float grad_output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        dX_out[idx] = (grad_output == 1.0f) ? dX_in[idx] : grad_output * dX_in[idx];
    }
}

void jsd_backward(
    const float* d_dX,  
    float* d_dX_out,     
    float grad_output,
    int total_elements)  
{
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    jsd_backward_kernel<<<blocks, threads>>>(d_dX, d_dX_out, grad_output, total_elements);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("jsd_backward_kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

__global__ void reduce_sum_kernel(const float* d_in, float* d_out, int N) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    float sum = 0.0f;
    if (idx < N) {
        sum = d_in[idx];
        if (idx + blockDim.x < N)
            sum += d_in[idx + blockDim.x];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        d_out[blockIdx.x] = sdata[0];
}

float reduce_loss(float* d_in, int N) {
    int threads = 256;
    int blocks = (N + threads * 2 - 1) / (threads * 2);
    float* d_partial;
    cudaMalloc(&d_partial, blocks * sizeof(float));

    int current_N = N;
    float sum = 0.0f;
    float* d_current = d_in;
    while (true) {
        size_t shmem_bytes = threads * sizeof(float);
        reduce_sum_kernel<<<blocks, threads, shmem_bytes>>>(d_current, d_partial, current_N);
        cudaDeviceSynchronize();

        if (blocks == 1) {
            cudaMemcpy(&sum, d_partial, sizeof(float), cudaMemcpyDeviceToHost);
            break;
        }
        current_N = blocks;
        blocks = (current_N + threads * 2 - 1) / (threads * 2);
        d_current = d_partial;
    }
    cudaFree(d_partial);
    return sum;
}

float cpu_jsd_forward(const float* h_input, const float* h_target, int BT, int V,
                      float beta, int ignore_index, bool has_label, int n_non_ignore,
                      float* h_dX)
{
    float total_loss = 0.0f;
    float scale = 1.0f / n_non_ignore;
    for (int row = 0; row < BT; row++) {
        float row_max = -INFINITY;
        for (int col = 0; col < V; col++) {
            float x_val = h_input[row * V + col];
            float y_val = h_target[row * V + col];
            row_max = std::max(row_max, std::max(x_val, y_val));
        }
        for (int col = 0; col < V; col++) {
            float x_val = h_input[row * V + col];
            float y_val = h_target[row * V + col];
            float l = 0.0f;
            float dx = 0.0f;
            if (beta == 0.0f) {
                float y_prob = std::exp(y_val); 
                l = y_prob * (y_val - x_val);
                dx = -y_prob;
            } else if (beta == 1.0f) {
                float x_prob = std::exp(x_val);
                l = x_prob * (x_val - y_val);
                dx = l + x_prob;
            } else {
                float x_shifted = x_val - row_max;
                float y_shifted = y_val - row_max;
                float exp_max = std::exp(row_max);
                float Q = std::exp(x_shifted) * exp_max;
                float P = std::exp(y_shifted) * exp_max;
                float beta_P = beta * P;
                float one_minus_beta_Q = (1.0f - beta) * Q;
                float M = beta_P + one_minus_beta_Q;
                float log_M = std::log(M);
                l = beta_P * y_val + one_minus_beta_Q * x_val - M * log_M;
                dx = one_minus_beta_Q * (x_val - log_M);
            }
            l *= scale;
            dx *= scale;
            total_loss += l;
            h_dX[row * V + col] = dx;
        }
    }
    return total_loss;
}

int main() {
    const int BT = 128;  
    const int V = 2048;  
    const int total = BT * V;
    float beta = 0.5f;
    int ignore_index = -100;
    bool has_label = false;
    int n_non_ignore = BT;

    float* h_input = new float[total];
    float* h_target = new float[total];
    float* h_dX_cpu = new float[total];
    for (int i = 0; i < total; i++) {
        h_input[i] = 0.1f;  
        h_target[i] = 0.2f; 
    }

    float *d_input, *d_target, *d_loss, *d_dX;
    cudaMalloc(&d_input, total * sizeof(float));
    cudaMalloc(&d_target, total * sizeof(float));
    cudaMalloc(&d_loss, total * sizeof(float));
    cudaMalloc(&d_dX, total * sizeof(float));

    cudaMemcpy(d_input, h_input, total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, h_target, total * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    jsd_forward(d_input, d_target, nullptr, beta, ignore_index, has_label, BT, V, d_loss, d_dX, n_non_ignore);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_ms = 0;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    float total_loss_gpu = reduce_loss(d_loss, total);

    auto cpu_start = std::chrono::high_resolution_clock::now();
    float total_loss_cpu = cpu_jsd_forward(h_input, h_target, BT, V, beta, ignore_index, has_label, n_non_ignore, h_dX_cpu);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;

    printf("GPU loss: %f, GPU time: %f ms\n", total_loss_gpu, gpu_ms);
    printf("CPU loss: %f, CPU time: %f ms\n", total_loss_cpu, cpu_duration.count());

    delete[] h_input;
    delete[] h_target;
    delete[] h_dX_cpu;
    cudaFree(d_input);
    cudaFree(d_target);
    cudaFree(d_loss);
    cudaFree(d_dX);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}