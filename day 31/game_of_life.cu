#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Define block dimensions.
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16

// CUDA kernel for one iteration of Game of Life using shared memory.
__global__ void gameOfLifeKernel(const int *in, int *out, int width, int height) {
    // Shared memory tile with a border: (blockDim.x+2) x (blockDim.y+2)
    extern __shared__ int sTile[];

    int bx = blockDim.x;
    int by = blockDim.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Global indices for the current thread.
    int globalX = blockIdx.x * bx + tx;
    int globalY = blockIdx.y * by + ty;
    
    // Shared memory tile indices (offset by 1 to leave room for halo).
    int sWidth = bx + 2;
    int sX = tx + 1;
    int sY = ty + 1;
    
    // Load the center cell.
    if (globalX < width && globalY < height)
        sTile[sY * sWidth + sX] = in[globalY * width + globalX];
    else
        sTile[sY * sWidth + sX] = 0;
    
    // Load halo cells.
    // Top halo
    if (ty == 0) {
        int gY = globalY - 1;
        int sY_top = 0;
        if (gY >= 0 && globalX < width)
            sTile[sY_top * sWidth + sX] = in[gY * width + globalX];
        else
            sTile[sY_top * sWidth + sX] = 0;
    }
    // Bottom halo
    if (ty == by - 1) {
        int gY = globalY + 1;
        int sY_bottom = by + 1;
        if (gY < height && globalX < width)
            sTile[sY_bottom * sWidth + sX] = in[gY * width + globalX];
        else
            sTile[sY_bottom * sWidth + sX] = 0;
    }
    // Left halo
    if (tx == 0) {
        int gX = globalX - 1;
        int sX_left = 0;
        if (gX >= 0 && globalY < height)
            sTile[sY * sWidth + sX_left] = in[globalY * width + gX];
        else
            sTile[sY * sWidth + sX_left] = 0;
    }
    // Right halo
    if (tx == bx - 1) {
        int gX = globalX + 1;
        int sX_right = bx + 1;
        if (gX < width && globalY < height)
            sTile[sY * sWidth + sX_right] = in[globalY * width + gX];
        else
            sTile[sY * sWidth + sX_right] = 0;
    }
    // Top-left corner
    if (tx == 0 && ty == 0) {
        int gX = globalX - 1;
        int gY = globalY - 1;
        int sIndex = 0; // Row 0, Col 0 in shared memory.
        if (gX >= 0 && gY >= 0)
            sTile[sIndex] = in[gY * width + gX];
        else
            sTile[sIndex] = 0;
    }
    // Top-right corner
    if (tx == bx - 1 && ty == 0) {
        int gX = globalX + 1;
        int gY = globalY - 1;
        int sIndex = (0 * sWidth) + (bx + 1);
        if (gX < width && gY >= 0)
            sTile[sIndex] = in[gY * width + gX];
        else
            sTile[sIndex] = 0;
    }
    // Bottom-left corner
    if (tx == 0 && ty == by - 1) {
        int gX = globalX - 1;
        int gY = globalY + 1;
        int sIndex = (by + 1) * sWidth + 0;
        if (gX >= 0 && gY < height)
            sTile[sIndex] = in[gY * width + gX];
        else
            sTile[sIndex] = 0;
    }
    // Bottom-right corner
    if (tx == bx - 1 && ty == by - 1) {
        int gX = globalX + 1;
        int gY = globalY + 1;
        int sIndex = (by + 1) * sWidth + (bx + 1);
        if (gX < width && gY < height)
            sTile[sIndex] = in[gY * width + gX];
        else
            sTile[sIndex] = 0;
    }
    
    // Ensure all shared memory loads are complete.
    __syncthreads();
    
    // Only process valid global cells.
    if (globalX < width && globalY < height) {
        // Count the number of live neighbors.
        int sum = 0;
        sum += sTile[(sY - 1) * sWidth + (sX - 1)];
        sum += sTile[(sY - 1) * sWidth + (sX)];
        sum += sTile[(sY - 1) * sWidth + (sX + 1)];
        sum += sTile[(sY) * sWidth + (sX - 1)];
        sum += sTile[(sY) * sWidth + (sX + 1)];
        sum += sTile[(sY + 1) * sWidth + (sX - 1)];
        sum += sTile[(sY + 1) * sWidth + (sX)];
        sum += sTile[(sY + 1) * sWidth + (sX + 1)];
        
        int cell = sTile[sY * sWidth + sX];
        int newState = 0;
        // Apply the Game of Life rules:
        //   - A live cell with 2 or 3 neighbors survives.
        //   - A dead cell with exactly 3 neighbors becomes alive.
        if (cell == 1 && (sum == 2 || sum == 3))
            newState = 1;
        else if (cell == 0 && sum == 3)
            newState = 1;
        else
            newState = 0;
        
        out[globalY * width + globalX] = newState;
    }
}

//
// Host code: setting up the grid, launching the kernel, and retrieving results.
//
int main() {
    // Set grid dimensions.
    const int width = 64;
    const int height = 64;
    const int size = width * height;
    
    // Allocate host memory.
    int *h_grid   = (int*)malloc(size * sizeof(int));
    int *h_result = (int*)malloc(size * sizeof(int));
    
    // Initialize the grid randomly (or you can set up a pattern like a glider).
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        h_grid[i] = rand() % 2;  // Random 0 or 1.
    }
    
    // Allocate device memory.
    int *d_grid, *d_result;
    cudaMalloc(&d_grid, size * sizeof(int));
    cudaMalloc(&d_result, size * sizeof(int));
    
    // Copy the initial grid from host to device.
    cudaMemcpy(d_grid, h_grid, size * sizeof(int), cudaMemcpyHostToDevice);
    
    // Define CUDA block and grid dimensions.
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    // Compute the size of shared memory needed.
    size_t sharedSize = (block.x + 2) * (block.y + 2) * sizeof(int);
    
    // Launch the Game of Life kernel (one iteration).
    gameOfLifeKernel<<<grid, block, sharedSize>>>(d_grid, d_result, width, height);
    cudaDeviceSynchronize();
    
    // Copy the updated grid back to host memory.
    cudaMemcpy(h_result, d_result, size * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Optionally, print the resulting grid.
    printf("Game of Life Grid After One Iteration:\n");
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            printf("%d ", h_result[y * width + x]);
        }
        printf("\n");
    }
    
    // Free device and host memory.
    cudaFree(d_grid);
    cudaFree(d_result);
    free(h_grid);
    free(h_result);
    
    return 0;
}
