#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <iostream>

enum ReductionMode {
    NONE = 0,
    SUM = 1,
    MEAN = 2,
    BATCHMEAN = 3
};

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                 \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(err);                                                        \
        }                                                                     \
    } while (0)

__global__ void tv_distance_kernel(
    const float* __restrict__ p,
    int p_stride,
    const float* __restrict__ q,
    int q_stride,
    float* __restrict__ loss,
    int loss_stride,
    float* __restrict__ grads,
    int grads_stride,
    const int* __restrict__ labels,
    int ignore_index,
    int n_cols,
    int reduction,
    bool hasLabel)
{
    int row = blockIdx.x;
    const float* p_row = p + row * p_stride;
    const float* q_row = q + row * q_stride;
    float* loss_row = loss + row * loss_stride;
    float* grads_row = grads + row * grads_stride;

    if (hasLabel) {
        int label = labels[row];
        if (label == ignore_index) {
            for (int col = threadIdx.x; col < n_cols; col += blockDim.x) {
                grads_row[col] = 0.0f;
                if (reduction == NONE) {
                    loss_row[col] = 0.0f;
                }
            }
            return;
        }
    }

    float thread_sum = 0.0f;
    for (int col = threadIdx.x; col < n_cols; col += blockDim.x) {
        float p_val = p_row[col];
        float q_val = q_row[col];
        float tv_loss = 0.5f * fabsf(p_val - q_val);
        float grad = (p_val > q_val) ? 0.5f : -0.5f;
        grads_row[col] = grad;

        if (reduction == NONE) {
            loss_row[col] = tv_loss;
        } else {
            thread_sum += tv_loss;
        }
    }

    if (reduction != NONE) {
        extern __shared__ float sdata[];
        int tid = threadIdx.x;
        sdata[tid] = thread_sum;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s)
                sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        if (tid == 0) {
            loss_row[0] = sdata[0];
        }
    }
}

void tv_distance_forward_cpu(const float* p, const float* q,
                             const int* labels, int ignore_index,
                             float* loss, float* grads,
                             int BT, int V,
                             ReductionMode reduction, bool hasLabel)
{
    for (int row = 0; row < BT; ++row) {
        bool ignore = hasLabel && (labels[row] == ignore_index);
        float row_sum = 0.0f;
        for (int col = 0; col < V; ++col) {
            if (ignore) {
                grads[row * V + col] = 0.0f;
                if (reduction == NONE)
                    loss[row * V + col] = 0.0f;
            } else {
                float p_val = p[row * V + col];
                float q_val = q[row * V + col];
                float tv_loss = 0.5f * fabsf(p_val - q_val);
                float grad = (p_val > q_val) ? 0.5f : -0.5f;
                grads[row * V + col] = grad;
                if (reduction == NONE)
                    loss[row * V + col] = tv_loss;
                else
                    row_sum += tv_loss;
            }
        }
        if (reduction != NONE && !ignore) {
            loss[row] = row_sum;
        }
    }
}

float post_process_loss(const float* loss, int BT, int V, int n_non_ignore, ReductionMode reduction) {
    float final_loss = 0.0f;
    if (reduction == SUM || reduction == BATCHMEAN || reduction == MEAN) {
        for (int row = 0; row < BT; ++row) {
            final_loss += loss[row];
        }
        if (reduction == BATCHMEAN) {
            final_loss /= n_non_ignore;
        } else if (reduction == MEAN) {
            final_loss /= (n_non_ignore * V);
        }
    }
    return final_loss;
}

void tvd_backward_cpu(const float* grad_output, const float* grads, float* grad_input, int size) {
    float scale = grad_output[0];
    for (int i = 0; i < size; ++i) {
        grad_input[i] = grads[i] * scale;
    }
}

int main() {
    const int BT = 1024;
    const int V = 1024;
    const int ignore_index = -100;
    const ReductionMode reduction = BATCHMEAN;
    const bool hasLabel = true;

    size_t size_mat = BT * V * sizeof(float);
    size_t size_vec = BT * sizeof(float);
    float* h_p = (float*)malloc(size_mat);
    float* h_q = (float*)malloc(size_mat);
    float* h_loss_cpu = (reduction == NONE ? (float*)malloc(size_mat) : (float*)malloc(size_vec));
    float* h_grads_cpu = (float*)malloc(size_mat);
    float* h_grads_back_cpu = (float*)malloc(size_mat);
    int* h_labels = (int*)malloc(BT * sizeof(int));

    for (int i = 0; i < BT * V; ++i) {
        h_p[i] = static_cast<float>(rand()) / RAND_MAX;
        h_q[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    int n_non_ignore = 0;
    for (int i = 0; i < BT; ++i) {
        h_labels[i] = (rand() % 10 == 0) ? ignore_index : i;
        if (h_labels[i] != ignore_index) n_non_ignore++;
    }

    float *d_p, *d_q, *d_loss, *d_grads;
    int* d_labels;
    CUDA_CHECK(cudaMalloc(&d_p, size_mat));
    CUDA_CHECK(cudaMalloc(&d_q, size_mat));
    size_t loss_size = (reduction == NONE ? size_mat : size_vec);
    CUDA_CHECK(cudaMalloc(&d_loss, loss_size));
    CUDA_CHECK(cudaMalloc(&d_grads, size_mat));
    CUDA_CHECK(cudaMalloc(&d_labels, BT * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_p, h_p, size_mat, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q, h_q, size_mat, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_labels, h_labels, BT * sizeof(int), cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocks = BT;
    size_t sharedMemSize = threadsPerBlock * sizeof(float);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    tv_distance_kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(
        d_p, V,
        d_q, V,
        d_loss, (reduction == NONE ? V : 1),
        d_grads, V,
        d_labels,
        ignore_index,
        V,
        reduction,
        hasLabel
    );
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_gpu = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_gpu, start, stop));

    if (reduction == NONE) {
        CUDA_CHECK(cudaMemcpy(h_loss_cpu, d_loss, size_mat, cudaMemcpyDeviceToHost));
    } else {
        CUDA_CHECK(cudaMemcpy(h_loss_cpu, d_loss, size_vec, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaMemcpy(h_grads_cpu, d_grads, size_mat, cudaMemcpyDeviceToHost));

    auto cpu_start = std::chrono::high_resolution_clock::now();
    tv_distance_forward_cpu(h_p, h_q, h_labels, ignore_index, h_loss_cpu, h_grads_cpu, BT, V, reduction, hasLabel);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_cpu = cpu_end - cpu_start;

    float grad_output = 1.0f;
    tvd_backward_cpu(&grad_output, h_grads_cpu, h_grads_back_cpu, BT * V);

    float final_loss = 0.0f;
    if (reduction != NONE) {
        final_loss = post_process_loss(h_loss_cpu, BT, V, n_non_ignore, reduction);
    }

    std::cout << "GPU kernel time: " << ms_gpu << " ms\n";
    std::cout << "CPU implementation time: " << ms_cpu.count() << " ms\n";
    if (reduction != NONE) {
        std::cout << "Final reduced loss: " << final_loss << "\n";
    }

    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_loss));
    CUDA_CHECK(cudaFree(d_grads));
    CUDA_CHECK(cudaFree(d_labels));
    free(h_p);
    free(h_q);
    free(h_loss_cpu);
    free(h_grads_cpu);
    free(h_grads_back_cpu);
    free(h_labels);

    return 0;
}
