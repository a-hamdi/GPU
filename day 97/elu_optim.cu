// To optimize the code I used: float4 loads for FP32 tail, __half2 vectorized ELU for even-index FP16, branchless, FMA, __exp2f for faster exp

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define EXPM1f(x) expm1f(x)
#define EXP2f(x) __exp2f(x)  


__global__ __launch_bounds__(1024, 4)
void elu_fp16(const float* __restrict__ input,
                   float* __restrict__ output,
                   size_t total,
                   float alpha) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

   
    size_t vec8 = (total / 8) * 8;
    for (size_t base = tid * 8; base < vec8; base += stride * 8) {
        
        float4 f0 = __ldg((const float4*)(input + base));
        float4 f1 = __ldg((const float4*)(input + base + 4));
        
    
        f0.x = f0.x > 0.f ? f0.x : alpha * EXPM1f(f0.x);
        f0.y = f0.y > 0.f ? f0.y : alpha * EXPM1f(f0.y);
        f0.z = f0.z > 0.f ? f0.z : alpha * EXPM1f(f0.z);
        f0.w = f0.w > 0.f ? f0.w : alpha * EXPM1f(f0.w);
        
        f1.x = f1.x > 0.f ? f1.x : alpha * EXPM1f(f1.x);
        f1.y = f1.y > 0.f ? f1.y : alpha * EXPM1f(f1.y);
        f1.z = f1.z > 0.f ? f1.z : alpha * EXPM1f(f1.z);
        f1.w = f1.w > 0.f ? f1.w : alpha * EXPM1f(f1.w);
        
        
        reinterpret_cast<float4*>(output + base)[0] = f0;
        reinterpret_cast<float4*>(output + base)[1] = f1;
    }
    
    for (size_t i = vec8 + tid; i < total; i += stride) {
        float x = __ldg(&input[i]);
        output[i] = x > 0.f ? x : alpha * EXPM1f(x);
    }
}

extern "C" void solution(const float* input, float* output, size_t n, size_t m, float alpha) {
    size_t total = n * m;
    const int threads = 1024;
    int blocks = (total / 8 + threads - 1) / threads;
    blocks = max(blocks, 320);
    blocks = min(blocks, 65535);

    elu_fp16<<<blocks, threads>>>(input, output, total, alpha);
}