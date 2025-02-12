#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cuComplex.h>

#define N 1024        // Number of spatial points
#define DX 0.01       // Spatial step
#define DT 5e-7       // Reduced time step for better stability
#define HBAR 1.0      // Planck's constant (normalized)
#define MASS 1.0      // Particle mass (normalized)
#define BLOCK_SIZE 256

using complexd = double2;

__host__ __device__ inline complexd make_complexd(double real, double imag) {
    return make_double2(real, imag);
}

__host__ __device__ inline complexd complex_add(complexd a, complexd b) {
    return make_double2(a.x + b.x, a.y + b.y);
}

__host__ __device__ inline complexd complex_sub(complexd a, complexd b) {
    return make_double2(a.x - b.x, a.y - b.y);
}

__host__ __device__ inline complexd complex_mul(double scalar, complexd a) {
    return make_double2(scalar * a.x, scalar * a.y);
}

__host__ __device__ inline complexd complex_mul_complex(complexd a, complexd b) {
    return make_double2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__host__ __device__ inline double complex_norm(complexd a) {
    return a.x * a.x + a.y * a.y;
}

__global__ void evolve_wavefunction(complexd *psi, complexd *psi_next, double *potential) {
    __shared__ complexd shared_psi[BLOCK_SIZE + 2];
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int local_idx = threadIdx.x + 1;

    if (i < N) {
        shared_psi[local_idx] = psi[i];
    }
    if (threadIdx.x == 0) {
        shared_psi[0] = (i > 0) ? psi[i - 1] : psi[i];
    }
    if (threadIdx.x == blockDim.x - 1) {
        shared_psi[local_idx + 1] = (i < N - 1) ? psi[i + 1] : psi[i];
    }
    __syncthreads();

    if (i > 0 && i < N - 1) {
        complexd laplacian = complex_sub(complex_add(shared_psi[local_idx - 1], shared_psi[local_idx + 1]),
                                          complex_mul(2.0, shared_psi[local_idx]));
        laplacian = complex_mul(1.0 / (DX * DX), laplacian);

        complexd i_hbar = make_complexd(0.0, HBAR);
        complexd term1 = complex_mul(DT / (2.0 * MASS), complex_mul_complex(i_hbar, laplacian));
        complexd term2a = complex_mul(potential[i], psi[i]);
        complexd term2 = complex_mul(DT / HBAR, complex_mul_complex(i_hbar, term2a));

        psi_next[i] = complex_add(complex_sub(psi[i], term1), term2);
    }
}

__global__ void normalize_wavefunction(complexd *psi, double *block_sums) {
    __shared__ double shared_norm[BLOCK_SIZE];
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    shared_norm[tid] = (i < N) ? complex_norm(psi[i]) : 0.0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_norm[tid] += shared_norm[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sums[blockIdx.x] = shared_norm[0];
    }
}

__global__ void sum_blocks(double *block_sums, int num_blocks, double *norm_factor) {
    __shared__ double shared_sums[BLOCK_SIZE];
    int tid = threadIdx.x;

    shared_sums[tid] = (tid < num_blocks) ? block_sums[tid] : 0.0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sums[tid] += shared_sums[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *norm_factor = shared_sums[0];
    }
}

__global__ void apply_normalization(complexd *psi, double norm_factor) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N && norm_factor > 0.0) {
        double scale = rsqrt(norm_factor);
        psi[i] = complex_mul(scale, psi[i]);
    }
}

int main() {
    complexd *d_psi, *d_psi_next;
    double *d_potential, *d_norm_factor, *d_block_sums;
    complexd h_psi[N];
    double h_potential[N], h_norm_factor = 0.0;
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    
    for (int i = 0; i < N; i++) {
        double x = (i - N / 2) * DX;
        double envelope = exp(-x * x);
        h_psi[i] = make_complexd(envelope * cos(5.0 * x), envelope * sin(5.0 * x));
        h_potential[i] = 0.5 * x * x;
    }

   
    cudaMalloc(&d_psi, N * sizeof(complexd));
    cudaMalloc(&d_psi_next, N * sizeof(complexd));
    cudaMalloc(&d_potential, N * sizeof(double));
    cudaMalloc(&d_norm_factor, sizeof(double));
    cudaMalloc(&d_block_sums, blocks * sizeof(double));

    cudaMemcpy(d_psi, h_psi, N * sizeof(complexd), cudaMemcpyHostToDevice);
    cudaMemcpy(d_potential, h_potential, N * sizeof(double), cudaMemcpyHostToDevice);

    int threads = BLOCK_SIZE;

    for (int t = 0; t < 1000; t++) {
        evolve_wavefunction<<<blocks, threads>>>(d_psi, d_psi_next, d_potential);
        cudaDeviceSynchronize();

        complexd *temp = d_psi;
        d_psi = d_psi_next;
        d_psi_next = temp;

        if (t % 100 == 0) {
            normalize_wavefunction<<<blocks, threads>>>(d_psi, d_block_sums);
            cudaDeviceSynchronize();

            cudaMemset(d_norm_factor, 0, sizeof(double));
            sum_blocks<<<1, threads>>>(d_block_sums, blocks, d_norm_factor);
            cudaDeviceSynchronize();

            cudaMemcpy(&h_norm_factor, d_norm_factor, sizeof(double), cudaMemcpyDeviceToHost);
            apply_normalization<<<blocks, threads>>>(d_psi, h_norm_factor);
            cudaDeviceSynchronize();
        }
    }

    cudaMemcpy(h_psi, d_psi, N * sizeof(complexd), cudaMemcpyDeviceToHost);

  
    cudaFree(d_psi);
    cudaFree(d_psi_next);
    cudaFree(d_potential);
    cudaFree(d_norm_factor);
    cudaFree(d_block_sums);

    for (int i = 0; i < N; i += N / 10) {
        std::cout << "x: " << (i - N / 2) * DX 
                  << " | Psi: (" << h_psi[i].x << ", " << h_psi[i].y << ")\n";
    }

    return 0;
}

