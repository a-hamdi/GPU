import os
import subprocess
import time
import torch

CUDA_KERNEL_SOURCE = """
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

__global__ void ddpm_update(float *x, float *eps, float *out, float alpha, float beta, float alpha_bar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float inv_sqrt_alpha = 1.0f / sqrtf(alpha);
        float scale_eps = beta / sqrtf(1.0f - alpha_bar);
        out[idx] = inv_sqrt_alpha * (x[idx] - scale_eps * eps[idx]);
    }
}

int main() {
    int n = 1024 * 1024 * 3; // Simulating shape (3, 1024, 1024)
    float alpha = 0.9f, beta = 0.1f, alpha_bar = 0.5f;
    
    float *h_x = (float*)malloc(n * sizeof(float));
    float *h_eps = (float*)malloc(n * sizeof(float));
    float *h_out = (float*)malloc(n * sizeof(float));
    
    for (int i = 0; i < n; i++) {
        h_x[i] = ((float)rand() / RAND_MAX) * 2 - 1;
        h_eps[i] = ((float)rand() / RAND_MAX) * 2 - 1;
    }
    
    float *d_x, *d_eps, *d_out;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_eps, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));

    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_eps, h_eps, n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(1024);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);

    cudaDeviceSynchronize();
    float ms;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++) {
        ddpm_update<<<gridSize, blockSize>>>(d_x, d_eps, d_out, alpha, beta, alpha_bar, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    
    printf("CUDA Kernel Time: %f ms\\n", ms / 1000.0);
    
    cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    free(h_x); free(h_eps); free(h_out);
    cudaFree(d_x); cudaFree(d_eps); cudaFree(d_out);

    return 0;
}
"""

CUDA_FILE = "ddpm_kernel.cu"
EXECUTABLE = "./ddpm_kernel"

# Step 1: Write CUDA code to file
with open(CUDA_FILE, "w") as f:
    f.write(CUDA_KERNEL_SOURCE)

# Step 2: Compile the CUDA code
print("Compiling CUDA Kernel...")
compile_cmd = f"nvcc -o {EXECUTABLE} {CUDA_FILE}"
os.system(compile_cmd)

# Step 3: Execute CUDA binary and extract time
print("Running CUDA Kernel...")
cuda_output = subprocess.check_output([EXECUTABLE]).decode("utf-8")
cuda_time = float(cuda_output.strip().split(":")[-1].strip().split()[0])
print(f"CUDA Kernel Time: {cuda_time:.4f} ms")

# Step 4: Run PyTorch implementation
def normal_update(x: torch.Tensor, epsilon_pred: torch.Tensor, alpha: float, beta: float, alpha_bar: float):
    inv_sqrt_alpha = 1 / torch.sqrt(torch.tensor(alpha, device=x.device))
    scale_eps = beta / torch.sqrt(torch.tensor(1 - alpha_bar, device=x.device))
    return inv_sqrt_alpha * (x - scale_eps * epsilon_pred)

device = "cuda" if torch.cuda.is_available() else "cpu"
shape = (3, 1024, 1024)
x = torch.randn(shape, device=device)
eps = torch.randn(shape, device=device)
alpha = 0.9
beta = 0.1
alpha_bar = 0.5

def benchmark_pytorch(iterations=1000):
    start_time = time.time()
    for _ in range(iterations):
        _ = normal_update(x, eps, alpha, beta, alpha_bar)
    elapsed = time.time() - start_time
    return (elapsed / iterations) * 1000.0

pytorch_time = benchmark_pytorch()
print(f"PyTorch Time: {pytorch_time:.4f} ms")

# Step 5: Compare results
print(f"CUDA Kernel Time: {cuda_time:.4f} ms vs PyTorch Time: {pytorch_time:.4f} ms")
print(f"Speedup: {pytorch_time / cuda_time:.2f}x")
