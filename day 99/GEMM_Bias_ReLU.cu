#include <cuda_runtime.h>

constexpr int TILE_M = 32;   
constexpr int TILE_B = 32;  
constexpr int TILE_K = 16;   

constexpr int THREADS_X = 16;  
constexpr int THREADS_Y = 16;  

static_assert(TILE_M  % (THREADS_X * 2) == 0, "TILE_M must be 2×THREADS_X");
static_assert(TILE_B  % (THREADS_Y * 2) == 0, "TILE_B must be 2×THREADS_Y");


#define REG_TILE_B 2
#define REG_TILE_M 2


__launch_bounds__(THREADS_X * THREADS_Y, 4)
__global__ void gemm_bias_relu(
    const float* __restrict__ A,   
    const float* __restrict__ W,   
    const float* __restrict__ b,   
          float*       C,         
    size_t B, size_t N, size_t M)
{
   
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

   
    float acc[REG_TILE_B][REG_TILE_M] = { {0.0f, 0.0f},
                                          {0.0f, 0.0f} };

   
    int tx = threadIdx.x;  
    int ty = threadIdx.y;  
    int local_m0 = 2 * tx; 
    int local_b0 = 2 * ty; 

 
    __shared__ float sA[TILE_B][TILE_K];
    __shared__ float sW[TILE_K][TILE_M];


    for (int k0 = 0; k0 < (int)N; k0 += TILE_K) {

        int flatId = ty * THREADS_X + tx;
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int idx = flatId + i * (THREADS_X*THREADS_Y);
            
            int a_row = idx / TILE_K;
            int a_col = idx % TILE_K;
            int global_row = block_row * TILE_B + a_row;
            int global_col = k0       + a_col;
            float a_val = 0.0f;
            if (global_row < (int)B && global_col < (int)N)
                a_val = __ldg(&A[ global_row * N + global_col ]);
            sA[a_row][a_col] = a_val;

            int w_id = idx;
            int w_row = w_id / TILE_M;    
            int w_col = w_id % TILE_M;    
            int global_m = block_col * TILE_M + w_col;
            int global_k = k0            + w_row;
            float w_val = 0.0f;
            if (global_m < (int)M && global_k < (int)N)
                w_val = __ldg(&W[ global_m * N + global_k ]);
            sW[w_row][w_col] = w_val;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            float a_reg[REG_TILE_B];
            a_reg[0] = sA[ local_b0 + 0 ][ kk ];
            a_reg[1] = sA[ local_b0 + 1 ][ kk ];
            float w_reg[REG_TILE_M];
            w_reg[0] = sW[ kk ][ local_m0 + 0 ];
            w_reg[1] = sW[ kk ][ local_m0 + 1 ];

            acc[0][0] += a_reg[0] * w_reg[0];
            acc[0][1] += a_reg[0] * w_reg[1];
            acc[1][0] += a_reg[1] * w_reg[0];
            acc[1][1] += a_reg[1] * w_reg[1];
        }
        __syncthreads();
    }

    int base_row = block_row * TILE_B;
    int base_col = block_col * TILE_M;
    #pragma unroll
    for (int i = 0; i < REG_TILE_B; ++i) {
        int r = base_row + local_b0 + i;
        if (r >= (int)B) continue;
        #pragma unroll
        for (int j = 0; j < REG_TILE_M; ++j) {
            int c = base_col + local_m0 + j;
            if (c >= (int)M) continue;
            float v = acc[i][j] + b[c];
            
            C[r * M + c] = v > 0.0f ? v : 0.0f;
        }
    }
}


extern "C" void solution(
    const float* A, const float* W, const float* b,
          float* C,
    size_t B, size_t N, size_t M)
{
    dim3 blockDim(THREADS_X, THREADS_Y);
    dim3 gridDim( (M + TILE_M - 1) / TILE_M,
                  (B + TILE_B - 1) / TILE_B );
    
    gemm_bias_relu<<<gridDim, blockDim>>>(A, W, b, C, B, N, M);
    
}
