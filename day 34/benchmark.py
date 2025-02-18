import subprocess
import numpy as np
import matplotlib.pyplot as plt

# Define test sizes
sizes = [10**i for i in range(1, 10)]  # 10, 100, 1000, ..., 1 billion
cpu_times = []
gpu_times = []

# Compile CPU and GPU programs
subprocess.run("mpicc -o vec_add_cpu vec_add_cpu.c -O3", shell=True, check=True)
subprocess.run("nvcc -o vec_add_gpu vec_add_gpu.cu -O3", shell=True, check=True)

for size in sizes:
    print(f"Running for size: {size}")
    
    # Run MPI CPU version
    result_cpu = subprocess.run(
        f"mpirun --oversubscribe --allow-run-as-root -np 4 ./vec_add_cpu {size}",
        shell=True, capture_output=True, text=True
    )
    
    # Debugging output
    print("CPU Output:", result_cpu.stdout)
    print("Error (if any):", result_cpu.stderr)
    
    output = result_cpu.stdout.strip().split()
    if len(output) >= 2:
        cpu_time = float(output[-2])  # Extract time if available
    else:
        print("Error: Unexpected CPU output format ->", result_cpu.stdout)
        cpu_time = float('inf')  # Assign a large value to indicate failure
    cpu_times.append(cpu_time)
    
    # Run CUDA GPU version
    result_gpu = subprocess.run(
        f"./vec_add_gpu {size}", shell=True, capture_output=True, text=True
    )
    
    # Debugging output
    print("GPU Output:", result_gpu.stdout)
    print("Error (if any):", result_gpu.stderr)
    
    output = result_gpu.stdout.strip().split()
    if len(output) >= 2:
        gpu_time = float(output[-2])  # Extract time if available
    else:
        print("Error: Unexpected GPU output format ->", result_gpu.stdout)
        gpu_time = float('inf')  # Assign a large value to indicate failure
    gpu_times.append(gpu_time)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(sizes, cpu_times, marker='o', label='CPU (MPI + Unrolling)')
plt.plot(sizes, gpu_times, marker='s', label='GPU (CUDA)')
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Input Size (log scale)")
plt.ylabel("Execution Time (ms, log scale)")
plt.legend()
plt.grid()
plt.title("Performance Comparison: CPU vs. GPU for Vector Addition")
plt.show()