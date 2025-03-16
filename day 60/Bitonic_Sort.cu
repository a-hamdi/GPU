#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 1024

__global__ void bitonic_sort_shared(float* d_array) {
    __shared__ float s_data[N];
    int tid = threadIdx.x;
    
    if (tid < N) {
        s_data[tid] = d_array[tid];
    }
    __syncthreads();
    
    for (int k = 2; k <= N; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = tid ^ j;
            if (ixj > tid) {
                if ((tid & k) == 0) {
                    if (s_data[tid] > s_data[ixj]) {
                        float temp = s_data[tid];
                        s_data[tid] = s_data[ixj];
                        s_data[ixj] = temp;
                    }
                } else {
                    if (s_data[tid] < s_data[ixj]) {
                        float temp = s_data[tid];
                        s_data[tid] = s_data[ixj];
                        s_data[ixj] = temp;
                    }
                }
            }
            __syncthreads();
        }
    }
    
    if (tid < N) {
        d_array[tid] = s_data[tid];
    }
}

void bitonic_sort(float* h_array) {
    float* d_array;
    size_t size = N * sizeof(float);
    
    cudaMalloc((void**)&d_array, size);
    cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice);
    
    bitonic_sort_shared<<<1, N>>>(d_array);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_array, d_array, size, cudaMemcpyDeviceToHost);
    cudaFree(d_array);
}

void print_array(float* array) {
    for (int i = 0; i < N; i++) {
        printf("%f ", array[i]);
    }
    printf("\n");
}

int main() {
    float h_array[N];

    for (int i = 0; i < N; i++) {
        h_array[i] = rand() % 1000 / 10.0;
    }

    printf("Unsorted array:\n");
    print_array(h_array);

    bitonic_sort(h_array);

    printf("Sorted array:\n");
    print_array(h_array);

    return 0;
}
