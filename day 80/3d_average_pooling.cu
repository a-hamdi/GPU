#include <cuda_runtime.h>

__global__ void avg_pool3d_kernel_optimized(const float* __restrict__ input,
                                             float* __restrict__ output,
                                             int H, int W, int D,
                                             int kernel_size, int stride, int padding,
                                             int H_out, int W_out, int D_out) {
    int out_i = blockIdx.x * blockDim.x + threadIdx.x;
    int out_j = blockIdx.y * blockDim.y + threadIdx.y;
    int out_k = blockIdx.z * blockDim.z + threadIdx.z;

    if (out_i < H_out && out_j < W_out && out_k < D_out) {
        int in_start_i = out_i * stride - padding;
        int in_start_j = out_j * stride - padding;
        int in_start_k = out_k * stride - padding;
        float sum = 0.0f;
        const int kernel_volume = kernel_size * kernel_size * kernel_size;

        #pragma unroll
        for (int m = 0; m < kernel_size; m++) {
            int cur_i = in_start_i + m;
            #pragma unroll
            for (int n = 0; n < kernel_size; n++) {
                int cur_j = in_start_j + n;
                #pragma unroll
                for (int o = 0; o < kernel_size; o++) {
                    int cur_k = in_start_k + o;
                    if (cur_i >= 0 && cur_i < H &&
                        cur_j >= 0 && cur_j < W &&
                        cur_k >= 0 && cur_k < D) {
                        int idx = cur_i * (W * D) + cur_j * D + cur_k;
                        sum += input[idx];
                    }
                }
            }
        }
        int out_idx = out_i * (W_out * D_out) + out_j * D_out + out_k;
        output[out_idx] = sum / static_cast<float>(kernel_volume);
    }
}

extern "C" void solution(const float* input, int kernel_size, int stride, int padding,
                           float* output, size_t H, size_t W, size_t D) {
    int H_int = static_cast<int>(H);
    int W_int = static_cast<int>(W);
    int D_int = static_cast<int>(D);

    int H_out = ((H_int + 2 * padding - kernel_size) / stride) + 1;
    int W_out = ((W_int + 2 * padding - kernel_size) / stride) + 1;
    int D_out = ((D_int + 2 * padding - kernel_size) / stride) + 1;

    dim3 blockDim(8, 8, 8);
    dim3 gridDim((H_out + blockDim.x - 1) / blockDim.x,
                 (W_out + blockDim.y - 1) / blockDim.y,
                 (D_out + blockDim.z - 1) / blockDim.z);

    avg_pool3d_kernel_optimized<<<gridDim, blockDim>>>(input, output,
                                                       H_int, W_int, D_int,
                                                       kernel_size, stride, padding,
                                                       H_out, W_out, D_out);
    cudaDeviceSynchronize();
}
