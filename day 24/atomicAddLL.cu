#include <stdio.h>

__device__ long long atomicAddLL(long long *addr, long long val) {
    unsigned long long *uaddr = (unsigned long long *)addr;
    unsigned long long old = *uaddr, assumed;
    do {
        assumed = old;
        old = atomicCAS(uaddr, assumed, assumed + val);
    } while (assumed != old);
    return (long long)old;
}

__global__ void atomicAddKernel(long long *data) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    atomicAddLL(data, tid);
}

int main() {
    long long *d_data;
    long long h_data = 0;

    cudaMalloc(&d_data, sizeof(long long));
    cudaMemcpy(d_data, &h_data, sizeof(long long), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = 4;
    
    // Correct kernel launch
    atomicAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_data, d_data, sizeof(long long), cudaMemcpyDeviceToHost);
    printf("Final value: %lld\n", h_data); // Expected: Sum of thread indices

    cudaFree(d_data);
    return 0;
}
