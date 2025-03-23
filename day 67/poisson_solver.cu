#include <iostream>
#include <cuda_runtime.h>

#define N 500  
#define BLOCK_SIZE 16  
#define TOLERANCE 1e-5 // Convergence threshold

__global__ void jacobi_kernel(double *u_new, double *u_old, double *f, int n, double h2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    if (i < n-1 && j < n-1) {
        int idx = j * n + i;
        u_new[idx] = 0.25 * (u_old[idx - 1] + u_old[idx + 1] + u_old[idx - n] + u_old[idx + n] - h2 * f[idx]);
    }
}

void solve_poisson(double *u, double *f, int n) {
    double *d_u_old, *d_u_new, *d_f;
    size_t size = n * n * sizeof(double);

    cudaMalloc(&d_u_old, size);
    cudaMalloc(&d_u_new, size);
    cudaMalloc(&d_f, size);

    cudaMemcpy(d_u_old, u, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u_new, u, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_f, f, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    double h2 = 1.0 / (n * n);
    int maxIter = 10000;
    for (int iter = 0; iter < maxIter; iter++) {
        jacobi_kernel<<<numBlocks, threadsPerBlock>>>(d_u_new, d_u_old, d_f, n, h2);
        cudaMemcpy(d_u_old, d_u_new, size, cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy(u, d_u_new, size, cudaMemcpyDeviceToHost);
    cudaFree(d_u_old);
    cudaFree(d_u_new);
    cudaFree(d_f);
}

int main() {
    double *u = new double[N * N]();
    double *f = new double[N * N]();

    // Set up the source term f(x,y)
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            f[j * N + i] = sin(i * M_PI / N) * sin(j * M_PI / N);
        }
    }

    solve_poisson(u, f, N);
    
    for (int j = 0; j < N; j += N / 16) { 
        for (int i = 0; i < N; i += N / 16) {
            std::cout << u[j * N + i] << " ";
        }
        std::cout << std::endl;
    }
    
    delete[] u;
    delete[] f;
    return 0;
}
