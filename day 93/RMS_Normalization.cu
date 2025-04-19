#include <cuda_runtime.h>
#include <math.h>

#define EPSILON 1e-5f

__global__ void compute_rms(const float* X, float* rms, size_t B, size_t N) {
    extern __shared__ float sdata[];
    size_t row = blockIdx.x;
    size_t tid = threadIdx.x;
    const float* row_ptr = X + row * N;

    float sum = 0.0f;
    for (size_t i = tid; i < N; i += blockDim.x) {
        float v = row_ptr[i];
        sum += v * v;
    }
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float mean_sq = sdata[0] / static_cast<float>(N);
        rms[row] = sqrtf(mean_sq + EPSILON);
    }
}

__global__ void normalize_rms(const float* X, float* Y, const float* rms, size_t B, size_t N) {
    size_t row = blockIdx.x;
    size_t tid = threadIdx.x;
    float r = rms[row];
    const float* row_in = X + row * N;
    float* row_out = Y + row * N;

    for (size_t i = tid; i < N; i += blockDim.x) {
        row_out[i] = row_in[i] / r;
    }
}

extern "C" void solution(const float* X, float* Y, size_t B, size_t N) {
    int threads = (N < 256) ? int(N) : 256;
    size_t shared_mem_size = threads * sizeof(float);

    float* d_rms = nullptr;
    cudaMalloc(&d_rms, B * sizeof(float));

    compute_rms<<<B, threads, shared_mem_size>>>(X, d_rms, B, N);

    normalize_rms<<<B, threads>>>(X, Y, d_rms, B, N);

    cudaFree(d_rms);
}
