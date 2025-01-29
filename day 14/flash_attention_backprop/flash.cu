#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>  
#include <limits> 
#include "helper.cuh"
#include "kernels.cuh"
#define BLOCK_SIZE 1024
#define THREADS_PER_BLOCK 1024
#define NEGATIVE_INFINITY -1e38f
void flashAttention2BackwardPass(const float* Q, const float* K, const float* V, const float* O, const float* dO, float* dQ, float* dK, float* dV, int N, int d, int Bc, int Br, float* Lhost) {
    float scale = 1.0f / sqrtf((float)d);
    // Initialize D
    float* D_device;
    cudaMalloc((void**)&D_device, N * sizeof(float));
    computeDKernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(dO, O, D_device, N, d);
    cudaDeviceSynchronize();

    float* D_host = (float*)malloc(N * sizeof(float));
    cudaMemcpy(D_host, D_device, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Initialize dQ, dK, dV on device
    cudaMemset(dQ, 0, N * d * sizeof(float));
    cudaMemset(dK, 0, N * d * sizeof(float));
    cudaMemset(dV, 0, N * d * sizeof(float));

    for (int j = 0; j < (N + Bc - 1) / Bc; ++j) {
        // Load Kj, Vj from HBM to on-chip SRAM (Simulated by host memory for now)
        float* Kj_host = (float*)malloc(Bc * d * sizeof(float));
        float* Vj_host = (float*)malloc(Bc * d * sizeof(float));
        cudaMemcpy(Kj_host, K + j * Bc * d, Bc * d * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(Vj_host, V + j * Bc * d, Bc * d * sizeof(float), cudaMemcpyDeviceToHost);

        float* Kj_device;
        float* Vj_device;
        cudaMalloc((void**)&Kj_device, Bc * d * sizeof(float));
        cudaMalloc((void**)&Vj_device, Bc * d * sizeof(float));
        cudaMemcpy(Kj_device, Kj_host, Bc * d * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(Vj_device, Vj_host, Bc * d * sizeof(float), cudaMemcpyHostToDevice);

        // Initialize dKj, dVj on SRAM (Simulated by device memory for now)
        float* dKj_temp;
        float* dVj_temp;
        cudaMalloc((void**)&dKj_temp, Bc * d * sizeof(float));
        cudaMalloc((void**)&dVj_temp, Bc * d * sizeof(float));
        cudaMemset(dKj_temp, 0, Bc * d * sizeof(float));
        cudaMemset(dVj_temp, 0, Bc * d * sizeof(float));

        for (int i = 0; i < (N + Br - 1) / Br; ++i) {
            // Load Qi, dOi, dQi, Li, Di from HBM to on-chip SRAM (Simulated by device memory for now)
            const float* Qi = Q + i * Br * d;
            const float* dOi = dO + i * Br * d;
            float* dQi_temp;
            cudaMalloc((void**)&dQi_temp, Br * d * sizeof(float));
            cudaMemset(dQi_temp, 0, Br * d * sizeof(float));

            const float* Li = Lhost + i * Br; // Assuming L is divided into blocks of size Br
            const float* Di = D_host + i * Br; // D is now divided into blocks of size Br

            // Allocate intermediate buffers on device for each loop iteration
            float* Si_device;
            float* Pi_device;
            float* dPi_device;
            float* dSi_device;
            cudaMalloc((void**)&Si_device, Br * Bc * sizeof(float));
            cudaMalloc((void**)&Pi_device, Br * Bc * sizeof(float));
            cudaMalloc((void**)&dPi_device, Br * Bc * sizeof(float));
            cudaMalloc((void**)&dSi_device, Br * Bc * sizeof(float));

            // Compute S_i
            computeSiKernel<<<(Br + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(Qi, Kj_device, Si_device, Br, Bc, d, scale);
            cudaDeviceSynchronize();

            // Find row-wise max of S_i
            float* maxSi_device;
            cudaMalloc((void**)&maxSi_device, Br * sizeof(float));
            findRowMaxSiKernel<<<(Br + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(Si_device, maxSi_device, Br, Bc);
            cudaDeviceSynchronize();

            // Compute P_i
            computePiKernel<<<(Br + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(Si_device, Li, Pi_device, Br, Bc, maxSi_device);
            cudaDeviceSynchronize();

            // Compute dVj += (P_i^T) * dOi
            computeDViKernel<<<(d + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(Pi_device, dOi, dVj_temp, Br, Bc, d);
            cudaDeviceSynchronize();

            // Compute dPi = dOi * V_j^T
            computeDPiKernel<<<(Br + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(dOi, Vj_device, dPi_device, Br, Bc, d);
            cudaDeviceSynchronize();

            // Compute dS_i = P_i * (dP_i - D_i)
            computeDSiKernel<<<(Br + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(Pi_device, dPi_device, Di, dSi_device, Br, Bc);
            cudaDeviceSynchronize();

            // Compute dQ_i += dS_i * K_j
            computeDQiKernel<<<(Br + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(dSi_device, Kj_device, dQi_temp, Br, d, Bc);
            cudaDeviceSynchronize();

            // Compute dKj += dS_i^T * Q_i
            computeDKjKernel<<<(Bc + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(dSi_device, Qi, dKj_temp, Bc, d, Br);
            cudaDeviceSynchronize();

            // Accumulate into dQ
            accumulateDQKernel<<<(Br * d + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(dQ, dQi_temp, Br, d, i * Br * d);
            cudaDeviceSynchronize();

            // Free intermediate buffers
            cudaFree(Si_device);
            cudaFree(Pi_device);
            cudaFree(dPi_device);
            cudaFree(dSi_device);
            cudaFree(maxSi_device);
            cudaFree(dQi_temp);
        }

        // Accumulate into dK and dV
        accumulateDKVjKernel<<<(Bc * d + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(dK, dV, dKj_temp, dVj_temp, Bc, d, j * Bc * d);
        cudaDeviceSynchronize();

        // Free device memory for Kj, Vj, dKj, dVj
        cudaFree(Kj_device);
        cudaFree(Vj_device);
        cudaFree(dKj_temp);
        cudaFree(dVj_temp);
        free(Kj_host);
        free(Vj_host);
    }

    cudaFree(D_device);
    free(D_host);
}