#ifndef BFS_H
#define BFS_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>

#define THREADS_PER_BLOCK 256
#define MAX_FRONTIER_SIZE 100000000
#define AVERAGE_EDGES_PER_VERTEX 8
#define NUM_VERTICES 100000000

#define CHECK_CUDA_ERROR(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error: %s (code %d) at %s:%d\n", \
                   cudaGetErrorString(err), err, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    }

void generate_random_graph(int num_vertices, int* num_edges, int** edges, int** dest);
void bfs_gpu(int source, int num_vertices, int num_edges, int* h_edges, int* h_dest, int* h_labels);
void cpu_bfs(int source, int num_vertices, int num_edges, int* edges, int* dest, int* label);

#endif // BFS_H
