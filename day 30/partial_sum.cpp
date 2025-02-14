#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void partialSumKernel(int *input, int *output, int n) {
    // Shared memory 
    HIP_DYNAMIC_SHARED(int, sharedMemory);
    
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;

    if (index < n/2) {  // Changed condition to handle half the array size
        // Load input into shared memory and optimize the loading to do coalescing 
        sharedMemory[tid] = input[2*index] + input[2*index + 1];
        __syncthreads();

        // Perform inclusive scan in shared memory
        for (int stride = 1; stride < blockDim.x; stride *= 2) {
            int temp = 0;
            if (tid >= stride) {
                temp = sharedMemory[tid - stride];
            }
            __syncthreads();
            sharedMemory[tid] += temp;
            __syncthreads();
        }

        // Write result to global memory
        output[index] = sharedMemory[tid];
    }
}

int main() {
    const int N = 16;
    const int blockSize = 8;

    int h_input[N] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    int h_output[N/2];  // Changed size to N/2 as we're summing pairs

    int *d_input, *d_output;
    size_t input_size = N * sizeof(int);
    size_t output_size = (N/2) * sizeof(int);

    hipMalloc(&d_input, input_size);
    hipMalloc(&d_output, output_size);

    hipMemcpy(d_input, h_input, input_size, hipMemcpyHostToDevice);

    // Launch kernel using hipLaunchKernelGGL
    hipLaunchKernelGGL(partialSumKernel,
                       dim3(1),  // Only need one block for this small example
                       dim3(blockSize),
                       blockSize * sizeof(int), 0,
                       d_input, d_output, N);
    
    hipDeviceSynchronize();

    hipMemcpy(h_output, d_output, output_size, hipMemcpyDeviceToHost);

    printf("Input: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_input[i]);
    }
    printf("\nOutput (sum of adjacent pairs): ");
    for (int i = 0; i < N/2; i++) {
        printf("%d ", h_output[i]);
    }
    printf("\n");

    hipFree(d_input);
    hipFree(d_output);

    return 0;
} 