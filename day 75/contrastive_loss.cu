#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

#define BATCH_SIZE 1024
#define DIM 128
#define MARGIN 1.0f


#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Euclidean Distance Kernel
__device__ float euclidean_distance(const float* x1, const float* x2, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = x1[i] - x2[i];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

// Contrastive Loss Kernel (Forward)
__global__ void contrastive_loss_forward(
    const float* x1, const float* x2, const int* labels, 
    float* loss, int batch_size, int dim, float margin) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        float dist = euclidean_distance(&x1[idx * dim], &x2[idx * dim], dim);
        int y = labels[idx];  // 1: similar, 0: dissimilar

        float term = fmaxf(0.0f, margin - dist);
        float loss_val = (1 - y) * 0.5f * dist * dist +
                         y * 0.5f * term * term;
        loss[idx] = loss_val;
    }
}

// CPU function to initialize random data
void initialize_data(float* x1, float* x2, int* labels, int batch_size, int dim) {
    for (int i = 0; i < batch_size * dim; i++) {
        x1[i] = static_cast<float>(rand()) / RAND_MAX;  
        x2[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < batch_size; i++) {
        labels[i] = rand() % 2;  // 0 or 1
    }
}


int main() {
    float *h_x1, *h_x2, *h_loss;
    int *h_labels;

    float *d_x1, *d_x2, *d_loss;
    int *d_labels;

    
    h_x1 = new float[BATCH_SIZE * DIM];
    h_x2 = new float[BATCH_SIZE * DIM];
    h_labels = new int[BATCH_SIZE];
    h_loss = new float[BATCH_SIZE];

    initialize_data(h_x1, h_x2, h_labels, BATCH_SIZE, DIM);

 
    CUDA_CHECK(cudaMalloc(&d_x1, BATCH_SIZE * DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x2, BATCH_SIZE * DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels, BATCH_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_loss, BATCH_SIZE * sizeof(float)));

  
    CUDA_CHECK(cudaMemcpy(d_x1, h_x1, BATCH_SIZE * DIM * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x2, h_x2, BATCH_SIZE * DIM * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_labels, h_labels, BATCH_SIZE * sizeof(int), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (BATCH_SIZE + threads - 1) / threads;
    contrastive_loss_forward<<<blocks, threads>>>(d_x1, d_x2, d_labels, d_loss, BATCH_SIZE, DIM, MARGIN);
    CUDA_CHECK(cudaDeviceSynchronize());

    
    CUDA_CHECK(cudaMemcpy(h_loss, d_loss, BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Sample Contrastive Loss Values:\n";
    for (int i = 0; i < 10; i++) {
        std::cout << "Loss[" << i << "]: " << h_loss[i] << std::endl;
    }

   
    delete[] h_x1;
    delete[] h_x2;
    delete[] h_labels;
    delete[] h_loss;

    CUDA_CHECK(cudaFree(d_x1));
    CUDA_CHECK(cudaFree(d_x2));
    CUDA_CHECK(cudaFree(d_labels));
    CUDA_CHECK(cudaFree(d_loss));

    return 0;
}
