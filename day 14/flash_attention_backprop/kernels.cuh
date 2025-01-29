#ifndef KERNELS_CUH
#define KERNELS_CUH

__global__ void computeDKernel(const float* dO, const float* O, float* D, int N, int d);

__global__ void computeSiKernel(const float* Qi, const float* Kj, float* Si, int Br, int Bc, int d, float scale);

__global__ void findRowMaxSiKernel(float* Si, float* maxSi, int Br, int Bc);

__global__ void computeSoftmaxKernel(float* Si, float* softmaxSi, int Br, int Bc);

__global__ void computeAttentionKernel(const float* Q, const float* K, const float* V, float* attention, int N, int d);

__global__ void computeQKernel(const float* Q, const float* dO, float* dQ, int N, int d);

__global__ void computeKKernel(const float* K, const float* dO, float* dK, int N, int d);

__global__ void computeVKernel(const float* V, const float* dO, float* dV, int N, int d);

__global__ void computeGradientsKernel(const float* dO, float* dQ, float* dK, float* dV, int N, int d);

#endif
