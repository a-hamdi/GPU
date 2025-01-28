#include "bfs.h"

__global__ void bfs_kernel(int level, int num_vertices, int* edges, int* dest, int* labels, int* done) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < num_vertices && labels[tid] == level) {
        for (int edge = edges[tid]; edge < edges[tid + 1]; edge++) {
            int neighbor = dest[edge];
            if (atomicCAS(&labels[neighbor], -1, level + 1) == -1) {
                atomicExch(done, 0);
            }
        }
    }
}
