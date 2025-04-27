#include <cuda_runtime.h>
#include <cstdio>

static constexpr float EPS = 1e-10f;

template <int UNROLL>
__global__ __launch_bounds__(256, 4)
void kl_divergence_kernel(const float* __restrict__ predictions,
                          const float* __restrict__ targets,
                          float* __restrict__ output,
                          size_t n)
{
    const size_t base = (size_t)blockIdx.x * blockDim.x * UNROLL + threadIdx.x;
    const size_t stride = (size_t)blockDim.x * gridDim.x * UNROLL;

    size_t idx = base;
#pragma unroll
    for (int i = 0; i < UNROLL; ++i, idx += blockDim.x) {
        if (idx < n) {
            // we load via read-only cache
            float t = __ldg(&targets[idx]) + EPS;
            float p = __ldg(&predictions[idx]) + EPS;
            // we compute fast-log diff
            float diff = __logf(t) - __logf(p);
            // we use fused multiply-add
            output[idx] = __fmaf_rn(t, diff, 0.0f);
        }
    }
}


extern "C" void solution(const float* predictions,
                         const float* targets,
                         float* output,
                         size_t n)
{
    const int threads = 256;
    const int UNROLL = 4;
    size_t elems_per_block = threads * UNROLL;
    int blocks = (int)((n + elems_per_block - 1) / elems_per_block);

    // We have to avoid oversubscription!!!
    blocks = min(blocks, 1024);

    kl_divergence_kernel<UNROLL><<<blocks, threads>>>(predictions, targets, output, n);

}
