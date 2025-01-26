import subprocess
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import psutil
import os

def get_memory_info():
    memory = psutil.virtual_memory()
    return memory.used / (1024 ** 3), memory.total / (1024 ** 3)

def estimate_memory_usage(N, M):
    nnz = (N * M) // 3
    memory_gb = (nnz * (2 * 8 + 4)) / (1024 ** 3)
    return memory_gb

def verify_results(cuda_output_file, torch_output, N):
    # Read CUDA results
    cuda_results = []
    try:
        with open(cuda_output_file, 'r',encoding="utf-8") as f:
            cuda_results = [float(line.strip()) for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading CUDA results: {e}")
        return False

    # Convert PyTorch results to a flattened list
    torch_results = torch_output.cpu().numpy().flatten().tolist()

    # Verify lengths
    if len(cuda_results) != N:
        print(f"CUDA results length mismatch: Expected {N}, Got {len(cuda_results)}")
        return False

    if len(torch_results) != N:
        print(f"PyTorch results length mismatch: Expected {N}, Got {len(torch_results)}")
        return False

    # Compare results with tolerance
    max_diff = 0
    max_relative_diff = 0
    tolerance = 1e-5

    for i, (cuda_val, torch_val) in enumerate(zip(cuda_results, torch_results)):
        abs_diff = abs(cuda_val - torch_val)
        max_diff = max(max_diff, abs_diff)

        # Calculate relative difference
        if abs(cuda_val) > 1e-10:  # Avoid division by zero
            relative_diff = abs_diff / abs(cuda_val)
            max_relative_diff = max(max_relative_diff, relative_diff)

        if abs_diff > tolerance:
            print(f"Mismatch at index {i}: CUDA = {cuda_val}, PyTorch = {torch_val}")
            print(f"Absolute difference: {abs_diff}")
            return False

    print(f"Results match within tolerance of {tolerance}")
    print(f"Maximum absolute difference: {max_diff}")
    print(f"Maximum relative difference: {max_relative_diff}")
    return True


def compile_cuda_program():
    compile_command = ["nvcc", "mainy.cu", "-o", "mainy"]
    subprocess.run(compile_command, check=True)

def create_sparse_matrix_and_vector(N, M):
    estimated_memory = estimate_memory_usage(N, M)
    _, total_memory = get_memory_info()
    if estimated_memory > total_memory * 0.7:
        raise MemoryError(f"Estimated memory usage ({estimated_memory:.2f} GB) exceeds safe limit")

    chunk_size = 1000000
    indices = []
    values = []

    for i in range(0, N, chunk_size // M):
        end_i = min(i + chunk_size // M, N)
        for j in range(M):
            for ii in range(i, end_i):
                if (ii + j) % 3 == 0:
                    indices.append([ii, j])
                    values.append(float(ii + j))

        if len(indices) > chunk_size:
            indices_tensor = torch.tensor(indices, dtype=torch.long).t()
            values_tensor = torch.tensor(values, dtype=torch.float32)
            if 'final_indices' not in locals():
                final_indices = indices_tensor
                final_values = values_tensor
            else:
                final_indices = torch.cat([final_indices, indices_tensor], dim=1)
                final_values = torch.cat([final_values, values_tensor])
            indices = []
            values = []

    if indices:
        indices_tensor = torch.tensor(indices, dtype=torch.long).t()
        values_tensor = torch.tensor(values, dtype=torch.float32)
        if 'final_indices' not in locals():
            final_indices = indices_tensor
            final_values = values_tensor
        else:
            final_indices = torch.cat([final_indices, indices_tensor], dim=1)
            final_values = torch.cat([final_values, values_tensor])

    A = torch.sparse_coo_tensor(final_indices, final_values, (N, M))
    X = torch.ones(M, 1, dtype=torch.float32)

    return A, X

def run_cuda_program(N, M):
    with open('main.cu', 'r') as file:
        content = file.read()

  

    content = content.replace('const int N = 1000;', f'const int N = {N};')
    content = content.replace('const int M = 1000;', f'const int M = {M};')
    content = content.replace('const int threshold = 700;',
                            f'const int threshold = {int(np.floor(N*0.7))};')

    with open('mainy.cu', 'w') as file:
        file.write(content)

    compile_cuda_program()
    result = subprocess.run(['./mainy'], capture_output=True, text=True)

    time_line = [line for line in result.stdout.split('\n')
                if 'CUDA kernel time:' in line][0]
    return float(time_line.split(':')[1].strip().split()[0])

def run_torch_program(N, M, num_iterations=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        times = []
        for _ in range(num_iterations):
            A, X = create_sparse_matrix_and_vector(N, M)
            A = A.to(device)
            X = X.to(device)

            A = A.coalesce()

            # First run for result verification
            output_torch = torch.sparse.mm(A, X)

            # Verify results
            #if not verify_results("cuda_results.txt", output_torch, N):
            #    print("WARNING: Results don't match!")

            # Warm-up run
            torch.cuda.synchronize()

        
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            output_torch = torch.sparse.mm(A, X)
            end.record()

            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        del A, X, output_torch
        torch.cuda.empty_cache()

        return np.mean(times) / 1000.0

    except Exception as e:
        print(f"Error in PyTorch implementation: {str(e)}")
        torch.cuda.empty_cache()
        return None

def main():
    sizes = [(10,10), (1000, 1000), (2000, 2000), (3000, 3000), (4000, 4000),
            (5000, 5000), (8000, 8000), (10000, 10000), (15000, 15000)]

    results = {
        'sizes': sizes,
        'cuda_times': [],
        'torch_times': [],
        'results_match': []
    }

    for N, M in sizes:
        print(f"\nTesting size {N}x{M}")
        print(f"Estimated memory usage: {estimate_memory_usage(N, M):.2f} GB")
        used_mem, total_mem = get_memory_info()
        print(f"Current memory usage: {used_mem:.2f} GB / {total_mem:.2f} GB")

        try:
            cuda_time = run_cuda_program(N, M)
            results['cuda_times'].append(cuda_time)
            print(f"Custom CUDA implementation time: {cuda_time:.6f} seconds")
        except Exception as e:
            print(f"CUDA implementation failed: {e}")
            results['cuda_times'].append(None)

        try:
            torch_time = run_torch_program(N, M)
            results['torch_times'].append(torch_time)
            if torch_time is not None:
                print(f"PyTorch Sparse implementation time: {torch_time:.6f} seconds")
        except Exception as e:
            print(f"PyTorch implementation failed: {e}")
            results['torch_times'].append(None)

        import gc
        gc.collect()
        torch.cuda.empty_cache()

   

if __name__ == "__main__":
    main()
