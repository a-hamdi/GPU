#include <cuda_runtime.h>



__global__ void leaky_relu_vec4_kernel(
    const float* __restrict__ input,
    float          alpha,
    float* __restrict__ output,
    size_t         total_vec4,
    size_t         total_floats)
{
    const float4* in4  = reinterpret_cast<const float4*>(input);
    float4*       out4 = reinterpret_cast<float4*>(output);

    // global thread index and total threads
    size_t idx    = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    // 1) Process in float4 chunks
    for (size_t v = idx; v < total_vec4; v += stride) {
        // I used __ldg for read-only cache
        float4 x = __ldg(&in4[v]);
        // And we use branchless LeakyReLU per lane
        x.x = fmaxf(x.x, alpha * x.x);
        x.y = fmaxf(x.y, alpha * x.y);
        x.z = fmaxf(x.z, alpha * x.z);
        x.w = fmaxf(x.w, alpha * x.w);
        out4[v] = x;
    }

    // 2) Handle any remaining floats
    size_t offset = total_vec4 * 4;
    for (size_t i = offset + idx; i < total_floats; i += stride) {
        float v = __ldg(&input[i]);
        output[i] = fmaxf(v, alpha * v);
    }
}


extern "C" void solution(
    const float* input,
    float         alpha,
    float*       output,
    size_t        M,
    size_t        N)
{
    size_t total_floats = M * N;
    size_t total_vec4   = total_floats / 4;

    constexpr int TPB = 256;
    int blocks = (total_vec4 + TPB - 1) / TPB;
    if (blocks < 1) blocks = 1;

    leaky_relu_vec4_kernel<<<blocks, TPB>>>(
        input, alpha, output,
        total_vec4, total_floats
    );
}
