#include <cuda_runtime.h>

__global__
void conv1d(const float* A,
            const float* B,
            float*       C,
            size_t       N,
            size_t       K)
{
    size_t i      = blockIdx.x * blockDim.x + threadIdx.x;
    int    radius = int(K/2);

    if (i < N) {
        float sum = 0.0f;
        for (int j = 0; j < int(K); ++j) {
            int idx = int(i) + j - radius;
            if (idx >= 0 && idx < int(N)) {
                sum += A[idx] * B[j];
            }
        }
        C[i] = sum;
    }
}

extern "C"
void solution(const float* A,
              const float* B,
              float*       C,
              size_t       N,
              size_t       K)
{
    int threads = 1024;
    int blocks  = int((N + threads - 1) / threads);

    conv1d<<<blocks, threads>>>(A, B, C, N, K);
}
