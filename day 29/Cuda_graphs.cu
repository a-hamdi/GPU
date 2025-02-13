#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_CUDA(call) do {                                          \
    cudaError_t err = call;                                           \
    if (err != cudaSuccess) {                                         \
        printf("CUDA Error at %s %d: %s\n", __FILE__, __LINE__,      \
               cudaGetErrorString(err));                              \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while(0)

const int N = 100000;  // Smaller data size
const int NUM_ITERATIONS = 10000;  // More iterations
const int BLOCK_SIZE = 256;

__global__ void matrixAdd(float* A, float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void matrixScale(float* A, float scalar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        A[idx] = A[idx] * scalar;
    }
}

__global__ void matrixSquare(float* A, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        A[idx] = A[idx] * A[idx];
    }
}

__global__ void matrixOffset(float* A, float offset, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        A[idx] = A[idx] + offset;
    }
}

void printTiming(const char* title, float ms) {
    printf("%s: %.3f ms\n", title, ms);
}

void verifyResults(float* h_A, float* h_B, float* h_C, float* h_verify, int n) {
    // Compute expected results on CPU
    for (int i = 0; i < n; i++) {
        float temp = h_A[i] + h_B[i];  // Add
        temp = temp * 2.0f;            // Scale
        temp = temp * temp;            // Square
        h_verify[i] = temp + 1.0f;     // Offset
    }

    // Compare with GPU results
    bool match = true;
    for (int i = 0; i < n; i++) {
        if (fabs(h_verify[i] - h_C[i]) > 1e-5) {
            match = false;
            printf("Mismatch at index %d: Expected %f, Got %f\n", 
                   i, h_verify[i], h_C[i]);
            break;
        }
    }
    if (match) {
        printf("Verification successful! All values match expected result.\n");
    }
}

int main() {
    float *h_A, *h_B, *h_C, *h_verify;
    float *d_A, *d_B, *d_C;
    size_t size = N * sizeof(float);

    // Allocate host memory
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);
    h_verify = (float*)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_A, size));
    CHECK_CUDA(cudaMalloc(&d_B, size));
    CHECK_CUDA(cudaMalloc(&d_C, size));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Create CUDA stream and events
    cudaStream_t stream;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Calculate grid dimensions
    int blocksPerGrid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Warmup
    for (int i = 0; i < 10; i++) {
        matrixAdd<<<blocksPerGrid, BLOCK_SIZE, 0, stream>>>(d_A, d_B, d_C, N);
        matrixScale<<<blocksPerGrid, BLOCK_SIZE, 0, stream>>>(d_C, 2.0f, N);
        matrixSquare<<<blocksPerGrid, BLOCK_SIZE, 0, stream>>>(d_C, N);
        matrixOffset<<<blocksPerGrid, BLOCK_SIZE, 0, stream>>>(d_C, 1.0f, N);
    }
    cudaStreamSynchronize(stream);

    // Traditional execution
    cudaEventRecord(start, stream);
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        matrixAdd<<<blocksPerGrid, BLOCK_SIZE, 0, stream>>>(d_A, d_B, d_C, N);
        matrixScale<<<blocksPerGrid, BLOCK_SIZE, 0, stream>>>(d_C, 2.0f, N);
        matrixSquare<<<blocksPerGrid, BLOCK_SIZE, 0, stream>>>(d_C, N);
        matrixOffset<<<blocksPerGrid, BLOCK_SIZE, 0, stream>>>(d_C, 1.0f, N);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printTiming("Without CUDA Graphs", milliseconds);

    // Copy results for verification
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Graph capture
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    matrixAdd<<<blocksPerGrid, BLOCK_SIZE, 0, stream>>>(d_A, d_B, d_C, N);
    matrixScale<<<blocksPerGrid, BLOCK_SIZE, 0, stream>>>(d_C, 2.0f, N);
    matrixSquare<<<blocksPerGrid, BLOCK_SIZE, 0, stream>>>(d_C, N);
    matrixOffset<<<blocksPerGrid, BLOCK_SIZE, 0, stream>>>(d_C, 1.0f, N);
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
    CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    // Graph execution
    cudaEventRecord(start, stream);
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printTiming("With CUDA Graphs", milliseconds);

    // Copy results and verify
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    verifyResults(h_A, h_B, h_C, h_verify, N);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_verify);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(graphExec);

    return 0;
}

