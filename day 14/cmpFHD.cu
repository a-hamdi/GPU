#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <random>
#include <iostream>

#define FHD_THREADS_PER_BLOCK 256
#define PI 3.14159265358979323846
#define CHUNK_SIZE 256

__constant__ float kx_c[CHUNK_SIZE], ky_c[CHUNK_SIZE], kz_c[CHUNK_SIZE];

__global__ void cmpFHd(float* rPhi, float* iPhi, float* phiMag,
                       float* x, float* y, float* z,
                       float* rMu, float* iMu, int M) {
    int n = blockIdx.x * FHD_THREADS_PER_BLOCK + threadIdx.x;

    float xn_r = x[n]; 
    float yn_r = y[n]; 
    float zn_r = z[n];

    float rFhDn_r = rPhi[n]; 
    float iFhDn_r = iPhi[n];

    for (int m = 0; m < M; m++) {
        float expFhD = 2 * PI * (kx_c[m] * xn_r + ky_c[m] * yn_r + kz_c[m] * zn_r);
        
        float cArg = __cosf(expFhD);
        float sArg = __sinf(expFhD);

        rFhDn_r += rMu[m] * cArg - iMu[m] * sArg;
        iFhDn_r += iMu[m] * cArg + rMu[m] * sArg;
    }

    rPhi[n] = rFhDn_r;
    iPhi[n] = iFhDn_r;
    phiMag[n] = sqrtf(rFhDn_r * rFhDn_r + iFhDn_r * iFhDn_r);
}

int main() {
    int N = 1024; // Define problem size
    int M = 1024; // Number of samples

    std::cout << "Starting program..." << std::endl;

    float *x, *y, *z, *rMu, *iMu, *rPhi, *iPhi, *phiMag;
    
    // Allocate memory and check for errors
    cudaError_t cudaStatus;
    // cudaMallocManaged takes care of memory allocation on both CPU and GPU
    cudaStatus = cudaMallocManaged(&x, N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaMallocManaged failed for x: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    cudaStatus = cudaMallocManaged(&y, N * sizeof(float));
    cudaStatus = cudaMallocManaged(&z, N * sizeof(float));
    cudaStatus = cudaMallocManaged(&rMu, M * sizeof(float));
    cudaStatus = cudaMallocManaged(&iMu, M * sizeof(float));
    cudaStatus = cudaMallocManaged(&rPhi, N * sizeof(float));
    cudaStatus = cudaMallocManaged(&iPhi, N * sizeof(float));
    cudaStatus = cudaMallocManaged(&phiMag, N * sizeof(float));

    std::cout << "Memory allocated successfully" << std::endl;

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    // Initialize input arrays with random values
    for (int i = 0; i < N; i++) {
        x[i] = dis(gen);
        y[i] = dis(gen);
        z[i] = dis(gen);
        rPhi[i] = 0.0f;
        iPhi[i] = 0.0f;
        phiMag[i] = 0.0f;
    }

    // Initialize rMu and iMu
    for (int i = 0; i < M; i++) {
        rMu[i] = dis(gen);
        iMu[i] = dis(gen);
    }

    std::cout << "Data initialized successfully" << std::endl;

    // Print some initial values
    std::cout << "\nInitial values:" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << "x[" << i << "] = " << x[i] << ", y[" << i << "] = " << y[i] << ", z[" << i << "] = " << z[i] << std::endl;
    }

    // Process data in chunks
    for (int i = 0; i < M / CHUNK_SIZE; i++) {
        std::cout << "\nProcessing chunk " << i + 1 << " of " << M / CHUNK_SIZE << std::endl;

        cudaStatus = cudaMemcpyToSymbol(kx_c, &x[i * CHUNK_SIZE], CHUNK_SIZE * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            std::cout << "cudaMemcpyToSymbol failed: " << cudaGetErrorString(cudaStatus) << std::endl;
            return 1;
        }

        cudaStatus = cudaMemcpyToSymbol(ky_c, &y[i * CHUNK_SIZE], CHUNK_SIZE * sizeof(float));
        cudaStatus = cudaMemcpyToSymbol(kz_c, &z[i * CHUNK_SIZE], CHUNK_SIZE * sizeof(float));

        // Launch kernel
        cmpFHd<<<N / FHD_THREADS_PER_BLOCK, FHD_THREADS_PER_BLOCK>>>(
            rPhi, iPhi, phiMag, x, y, z, rMu, iMu, CHUNK_SIZE);
        
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            std::cout << "Kernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
            return 1;
        }

        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            std::cout << "cudaDeviceSynchronize failed: " << cudaGetErrorString(cudaStatus) << std::endl;
            return 1;
        }
    }

    std::cout << "\nComputation completed. Results:" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << "rPhi[" << i << "] = " << rPhi[i] 
                 << ", iPhi[" << i << "] = " << iPhi[i] 
                 << ", phiMag[" << i << "] = " << phiMag[i] << std::endl;
    }

    // Free memory
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
    cudaFree(rMu);
    cudaFree(iMu);
    cudaFree(rPhi);
    cudaFree(iPhi);
    cudaFree(phiMag);

    std::cout << "\nProgram completed successfully" << std::endl;
    return 0;
}
