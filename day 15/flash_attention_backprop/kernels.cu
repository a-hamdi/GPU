#include <cuda_runtime.h>
#include <math.h>
#include "kernels.cuh"


__global__ void computeDKernel(const float* dO, const float* O, float* D, int N, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float sum = 0.0f;
    for (int i = 0; i < d; i++) {
        sum += dO[idx * d + i] * O[idx * d + i];
    }
    D[idx] = sum;
}

__global__ void computeSiKernel(const float* Qi, const float* Kj, float* Si, int Br, int Bc, int d, float scale) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= Br) return;

    for (int col = 0; col < Bc; ++col) {
        float sum = 0.0f;
        for (int k = 0; k < d; ++k) {
            sum += Qi[row * d + k] * Kj[col * d + k];
        }
        Si[row * Bc + col] = sum * scale; // Apply scaling here
    }
}

__global__ void findRowMaxSiKernel(const float* Si, float* maxSi, int Br, int Bc) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= Br) return;

    __shared__ float shared_max[BLOCK_SIZE];
    float local_max = NEGATIVE_INFINITY;

    for (int col = threadIdx.x; col < Bc; col += blockDim.x) {
        local_max = fmaxf(local_max, Si[row * Bc + col]);
    }

    shared_max[threadIdx.x] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        maxSi[row] = shared_max[0];
    }
}

__global__ void computePiKernel(const float* Si, const float* Li, float* Pi, int Br, int Bc, const float* maxSi) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = threadIdx.x;
    if (row >= Br) return;

    __shared__ float shared_max[BLOCK_SIZE];
     if (col < Bc) {
         float si_val = Si[row * Bc + col];
         float li_val = Li[row];
         float max_si_val = maxSi[row];
         float val = expf(si_val - li_val - max_si_val);
         if (isnan(val) || isinf(val)) {
             val = 0.0f;
         }
         Pi[row * Bc + col] = val;
    }
}

__global__ void computeDViKernel(const float* Pi, const float* dOi, float* dVj_temp, int Br, int Bc, int d) {
    int col_dVi = blockIdx.x * blockDim.x + threadIdx.x;
    if (col_dVi >= d) return;

    for (int row_dVj = 0; row_dVj < Bc; ++row_dVj) {
        float sum = 0.0f;
        for (int row_Pi = 0; row_Pi < Br; ++row_Pi) {
            sum += Pi[row_Pi * Bc + row_dVj] * dOi[row_Pi * d + col_dVi];
        }
        dVj_temp[row_dVj * d + col_dVi] = sum;
    }
}

__global__ void computeDPiKernel(const float* dOi, const float* Vj, float* dPi, int Br, int Bc, int d) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= Br) return;

    for (int col = 0; col < Bc; ++col) {
        float sum = 0.0f;
        for (int k = 0; k < d; ++k) {
            sum += dOi[row * d + k] * Vj[col * d + k];
        }
        dPi[row * Bc + col] = sum;
    }
}

__global__ void computeDSiKernel(const float* Pi, const float* dPi, const float* Di, float* dSi, int Br, int Bc) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = threadIdx.x;

    if (row >= Br || col >= Bc) return;
    __shared__ float shared_di[BLOCK_SIZE];
     if(threadIdx.x < Bc) {
         shared_di[threadIdx.x] = Di[row]; // Load each element of Di
     }
     __syncthreads();
    dSi[row * Bc + col] = Pi[row * Bc + col] * (dPi[row * Bc + col] - shared_di[threadIdx.x]);
}

__global__ void computeDQiKernel(const float* dSi, const float* Kj, float* dQi_temp, int Br, int d, int Bc) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= Br) return;

    for (int col = 0; col < d; ++col) {
        float sum = 0.0f;
        for (int k = 0; k < Bc; ++k) {
            sum += dSi[row * Bc + k] * Kj[k * d + col];
        }
        dQi_temp[row * d + col] = sum;
    }
}

__global__ void computeDKjKernel(const float* dSi, const float* Qi, float* dKj_temp, int Bc, int d, int Br) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= Bc) return;

    for (int row = 0; row < d; ++row) {
        float sum = 0.0f;
        for (int k = 0; k < Br; ++k) {
            sum += dSi[k * Bc + col] * Qi[k * d + row];
        }
        dKj_temp[col * d + row] = sum;
    }
}

__global__ void accumulateDQKernel(float* dQ, const float* dQi_temp, int Br, int d, int globalOffset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Br * d) return;

    atomicAdd(&dQ[globalOffset + idx], dQi_temp[idx]);
}

__global__ void accumulateDKVjKernel(float* dK, float* dV, const float* dKj_temp, const float* dVj_temp, int Bc, int d, int globalOffset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Bc * d) return;

    atomicAdd(&dK[globalOffset + idx], dKj_temp[idx]);
    atomicAdd(&dV[globalOffset + idx], dVj_temp[idx]);
}