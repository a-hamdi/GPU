#include "solve.h"
#include <cuda_runtime.h>

// Each agent has [x, y, vx, vy] in agents array, for a total of 4*N floats.
// This kernel performs one flocking step for each agent.
__global__
void flockKernel(const float* agents, float* agents_next, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    
    // Current agent's data starts at index base = 4*i
    int base = 4 * i;
    float x  = agents[base + 0];
    float y  = agents[base + 1];
    float vx = agents[base + 2];
    float vy = agents[base + 3];
    
    // Flocking parameters
    const float r      = 5.0f;       // neighbor radius
    const float r_sq   = r * r;      // compare squared distances
    const float alpha  = 0.05f;      // alignment factor

    // Accumulate neighbor velocities
    float sum_vx = 0.0f;
    float sum_vy = 0.0f;
    int neighborCount = 0;
    
    // Find neighbors within radius r
    for (int j = 0; j < N; j++)
    {
        if (j == i) continue;  // skip self

        int jbase = 4 * j;
        float xj  = agents[jbase + 0];
        float yj  = agents[jbase + 1];
        
        float dx = xj - x;
        float dy = yj - y;
        float dist_sq = dx*dx + dy*dy;
        
        // "within radius" means dist < r, so dist_sq < r_sq
        if (dist_sq < r_sq) {
            // accumulate neighbor velocities
            sum_vx += agents[jbase + 2];
            sum_vy += agents[jbase + 3];
            neighborCount++;
        }
    }
    
    // Compute new velocity
    float new_vx = vx;
    float new_vy = vy;
    if (neighborCount > 0)
    {
        float avg_vx = sum_vx / neighborCount;
        float avg_vy = sum_vy / neighborCount;
        // v_new = v + alpha*(avg_v - v)
        new_vx = vx + alpha * (avg_vx - vx);
        new_vy = vy + alpha * (avg_vy - vy);
    }
    
    // Update position
    float new_x = x + new_vx;
    float new_y = y + new_vy;
    
    // Write results
    agents_next[base + 0] = new_x;
    agents_next[base + 1] = new_y;
    agents_next[base + 2] = new_vx;
    agents_next[base + 3] = new_vy;
}

// The solve function with default C++ linkage
void solve(const float* agents, float* agents_next, int N)
{
    // Device pointers
    float *d_agents     = nullptr;
    float *d_agentsNext = nullptr;
    
    size_t size = 4 * N * sizeof(float);
    
    // Allocate device memory
    cudaMalloc((void**)&d_agents,     size);
    cudaMalloc((void**)&d_agentsNext, size);
    
    // Copy agents from host to device
    cudaMemcpy(d_agents, agents, size, cudaMemcpyHostToDevice);
    
    // Configure kernel
    int blockSize = 256;
    int gridSize  = (N + blockSize - 1) / blockSize;
    
    // Launch kernel
    flockKernel<<<gridSize, blockSize>>>(d_agents, d_agentsNext, N);
    cudaDeviceSynchronize();
    
    // Copy results back to host
    cudaMemcpy(agents_next, d_agentsNext, size, cudaMemcpyDeviceToHost);
    
    // Clean up
    cudaFree(d_agents);
    cudaFree(d_agentsNext);
}
