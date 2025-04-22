#include <cuda_runtime.h>
#include <algorithm>


__global__
void prod_reduce_kernel(const float* __restrict__ input,
                        float* __restrict__ output,
                        size_t M,
                        size_t S_d,
                        size_t N)
{

    size_t out_idx = blockIdx.x;

    size_t m = out_idx / N;
    size_t n = out_idx - m * N;


    const float* base = input + (m * S_d) * N + n;


    double prod = 1.0;
    for (size_t k = threadIdx.x; k < S_d; k += blockDim.x) {
        prod *= static_cast<double>( base[k * N] );
    }


    constexpr unsigned FULL_MASK = 0xffffffffu;
    for (int offset = warpSize/2; offset > 0; offset >>= 1) {
        prod *= __shfl_down_sync(FULL_MASK, prod, offset);
    }

  
    __shared__ double warp_prod[1024/32];
    int lane = threadIdx.x & (warpSize - 1);
    int wid  = threadIdx.x >> 5;                 
    if (lane == 0) warp_prod[wid] = prod;
    __syncthreads();

  
    if (wid == 0) {
        double block_prod = (lane < ((blockDim.x+31)/32)) 
                            ? warp_prod[lane] 
                            : 1.0;
        for (int offset = ((blockDim.x+31)/32)/2; offset > 0; offset >>= 1) {
            block_prod *= __shfl_down_sync(FULL_MASK, block_prod, offset);
        }
        if (lane == 0) {
           
            output[out_idx] = static_cast<float>(block_prod);
        }
    }
}


extern "C"
void solution(const float*  input,
              int           dim,
              float*        output,
              size_t*       shape,    
              size_t        ndim)
{

    std::vector<size_t> hshape(ndim);
    cudaMemcpy(hshape.data(), shape, ndim*sizeof(size_t),
               cudaMemcpyDeviceToHost);


    size_t M = 1, N = 1;
    for (int i = 0; i < dim; ++i)         M *= hshape[i];
    for (int i = dim+1; i < (int)ndim; ++i) N *= hshape[i];
    size_t S_d = hshape[dim];

    size_t total_outputs = M * N;
    if (total_outputs == 0 || S_d == 0) return;


    int blk = 1;
    while (blk < (int)S_d && blk < 1024) blk <<= 1;
    blk = std::max(blk, 32);
    blk = std::min(blk, 1024);


    dim3 grid( total_outputs );
    dim3 block( blk );

    prod_reduce_kernel<<<grid,block>>>(input, output, M, S_d, N);

   
}
