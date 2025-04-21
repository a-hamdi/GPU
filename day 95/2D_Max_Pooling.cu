#include <cuda_runtime.h>
#include <float.h>    // for FLT_MAX
#include <stddef.h>   // for size_t


__global__
void maxpool2d_kernel(const float* __restrict__ input,
                      int H, int W,
                      int kernel_size, int stride, int padding, int dilation,
                      int H_out, int W_out,
                      float* __restrict__ output)
{

    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_y >= H_out || out_x >= W_out) return;


    float max_val = -FLT_MAX;
    for (int m = 0; m < kernel_size; ++m) {
        int in_y = out_y * stride + m * dilation - padding;
        for (int n = 0; n < kernel_size; ++n) {
            int in_x = out_x * stride + n * dilation - padding;
            
            if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                float v = input[in_y * W + in_x];
                if (v > max_val) max_val = v;
            }
        }
    }
    output[out_y * W_out + out_x] = max_val;
}


extern "C"
void solution(const float* input,
              int kernel_size,
              int stride,
              int padding,
              int dilation,
              float* output,
              size_t H,
              size_t W)
{

    int H_out = (int)(( (int)H + 2*padding
                       - dilation*(kernel_size-1)
                       - 1 ) / stride) + 1;
    int W_out = (int)(( (int)W + 2*padding
                       - dilation*(kernel_size-1)
                       - 1 ) / stride) + 1;


    const int Bx = 16, By = 16;
    dim3 block(Bx, By);
    dim3 grid( (W_out + Bx - 1) / Bx,
               (H_out + By - 1) / By );

    maxpool2d_kernel<<<grid, block>>>(
        input,
        (int)H, (int)W,
        kernel_size, stride, padding, dilation,
        H_out, W_out,
        output
    );


}
