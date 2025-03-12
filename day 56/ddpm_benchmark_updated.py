import os
import subprocess
import torch
import matplotlib.pyplot as plt

# --- CUDA Kernel Code (saved to file) ---
CUDA_KERNEL_SOURCE = r'''
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

__global__ void ddpm_update(float *x, float *eps, float *out, float alpha, float beta, float alpha_bar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float inv_sqrt_alpha = 1.0f / sqrtf(alpha);
        float scale_eps = beta / sqrtf(1.0f - alpha_bar);
        out[idx] = inv_sqrt_alpha * (x[idx] - scale_eps * eps[idx]);
    }
}

int main() {
    int n = 1024 * 1024 * 3; // shape: (3,1024,1024)
    float alpha = 0.9f, beta = 0.1f, alpha_bar = 0.5f;
    
    // Allocate host memory once.
    float *h_x = (float*)malloc(n * sizeof(float));
    float *h_eps = (float*)malloc(n * sizeof(float));
    float *h_out = (float*)malloc(n * sizeof(float));
    
    // Allocate device memory once.
    float *d_x, *d_eps, *d_out;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_eps, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    
    dim3 blockSize(1024);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
    
    // Setup CUDA events for timing.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warm-up iterations (not timed)
    int warmup = 10;
    srand(time(NULL));
    for (int iter = 0; iter < warmup; iter++) {
        for (int i = 0; i < n; i++) {
            h_x[i] = ((float)rand() / RAND_MAX) * 2 - 1;
            h_eps[i] = ((float)rand() / RAND_MAX) * 2 - 1;
        }
        cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_eps, h_eps, n * sizeof(float), cudaMemcpyHostToDevice);
        ddpm_update<<<gridSize, blockSize>>>(d_x, d_eps, d_out, alpha, beta, alpha_bar, n);
        cudaDeviceSynchronize();
    }
    
    // Timed iterations.
    int iterations = 1000;
    float total_time = 0.0f;
    for (int iter = 0; iter < iterations; iter++) {
        // Randomize inputs outside the timed section.
        for (int i = 0; i < n; i++) {
            h_x[i] = ((float)rand() / RAND_MAX) * 2 - 1;
            h_eps[i] = ((float)rand() / RAND_MAX) * 2 - 1;
        }
        cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_eps, h_eps, n * sizeof(float), cudaMemcpyHostToDevice);
        
        cudaEventRecord(start);
        ddpm_update<<<gridSize, blockSize>>>(d_x, d_eps, d_out, alpha, beta, alpha_bar, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_time += ms;
    }
    
    float avg_time = total_time / iterations;
    printf("CUDA Kernel Time: %f ms\n", avg_time);
    
    // (Optional) Copy back result.
    cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    free(h_x); free(h_eps); free(h_out);
    cudaFree(d_x); cudaFree(d_eps); cudaFree(d_out);
    
    return 0;
}
'''

CUDA_FILE = "ddpm_kernel.cu"
EXECUTABLE = "./ddpm_kernel"

# --- Write CUDA code to file ---
with open(CUDA_FILE, "w") as f:
    f.write(CUDA_KERNEL_SOURCE)

# --- Compile the CUDA code ---
print("Compiling CUDA Kernel...")
compile_cmd = f"nvcc -O2 -o {EXECUTABLE} {CUDA_FILE}"
os.system(compile_cmd)

# --- Run the CUDA binary and extract time ---
print("Running CUDA Kernel benchmark...")
cuda_output = subprocess.check_output([EXECUTABLE]).decode("utf-8")
# Expected output format: "CUDA Kernel Time: <time> ms"
cuda_time = float(cuda_output.strip().split(":")[-1].strip().split()[0])
print(f"CUDA Kernel Time: {cuda_time:.4f} ms")

# --- Revised PyTorch Implementation (using CUDA events for fair timing) ---
def normal_update_inplace(x: torch.Tensor, eps: torch.Tensor, inv_sqrt_alpha: torch.Tensor, scale_eps: torch.Tensor):
    return inv_sqrt_alpha * (x - scale_eps * eps)

def benchmark_pytorch_inplace(iterations=1000, shape=(3,1024,1024), alpha=0.9, beta=0.1, alpha_bar=0.5, device="cuda"):
    inv_sqrt_alpha = 1 / torch.sqrt(torch.tensor(alpha, device=device))
    scale_eps = beta / torch.sqrt(torch.tensor(1 - alpha_bar, device=device))
    
    # Preallocate tensors outside the loop.
    x = torch.empty(shape, device=device)
    eps = torch.empty(shape, device=device)
    
    # Create CUDA events for timing.
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Warm-up iterations (not timed)
    for _ in range(10):
        x.copy_(torch.randn(shape, device=device))
        eps.copy_(torch.randn(shape, device=device))
        _ = normal_update_inplace(x, eps, inv_sqrt_alpha, scale_eps)
        torch.cuda.synchronize()
    
    total_time = 0.0
    for _ in range(iterations):
        # Randomize inputs outside the timed section.
        x.copy_(torch.randn(shape, device=device))
        eps.copy_(torch.randn(shape, device=device))
        
        torch.cuda.synchronize()
        start_event.record()
        
        _ = normal_update_inplace(x, eps, inv_sqrt_alpha, scale_eps)
        
        end_event.record()
        torch.cuda.synchronize()
        # elapsed_time returns milliseconds.
        total_time += start_event.elapsed_time(end_event)
    
    avg_time = total_time / iterations
    return avg_time

device = "cuda" if torch.cuda.is_available() else "cpu"
pytorch_time = benchmark_pytorch_inplace(
    iterations=1000,
    shape=(3,1024,1024),
    alpha=0.9,
    beta=0.1,
    alpha_bar=0.5,
    device=device
)
print(f"PyTorch Time (isolated update): {pytorch_time:.4f} ms")

# --- Plot histogram comparing CUDA and PyTorch timings ---
methods = ["CUDA Kernel", "PyTorch (isolated update)"]
times = [cuda_time, pytorch_time]

plt.figure(figsize=(6, 4))
bars = plt.bar(methods, times, color=["blue", "orange"])
plt.xlabel("Method")
plt.ylabel("Execution Time (ms)")
plt.title("DDPM Update: CUDA vs PyTorch (isolated update timing)")
plt.yscale("log")
plt.grid(axis="y", linestyle="--", alpha=0.7)

for bar, t in zip(bars, times):
    plt.text(bar.get_x() + bar.get_width()/2, t * 1.05, f"{t:.4f} ms", ha="center", fontsize=10)

plt.show()

# --- Print speedup ratio ---
print(f"Speedup: {pytorch_time / cuda_time:.2f}x")
