
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// Reduction modes.
enum Reduction { NONE = 0, SUM = 1, MEAN = 2, BATCHMEAN = 3 };

// -------------------------------------------------------------------
// Forward kernels
// -------------------------------------------------------------------

__global__ void kldiv_forward_kernel_none(const float* __restrict__ y_pred,
                                            const float* __restrict__ y_true,
                                            float* __restrict__ loss,
                                            int V,
                                            float eps,
                                            bool log_target) {
    int b = blockIdx.x;  
    int i = threadIdx.x;
    int offset = b * V + i;
    if (i < V) {
        float pred = y_pred[offset];
        float target = y_true[offset];
        float val = 0.0f;
        if (!log_target) {
            
            val = target * (logf(fmaxf(target, eps)) - pred);
        } else {
            val = expf(target) * (target - pred);
        }
        loss[offset] = val;
    }
}


__global__ void kldiv_forward_kernel_reduce(const float* __restrict__ y_pred,
                                              const float* __restrict__ y_true,
                                              float* __restrict__ loss,
                                              int V,
                                              float eps,
                                              bool log_target) {
    int b = blockIdx.x; 
    extern __shared__ float sdata[]; 
    float sum = 0.0f;

   
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        int offset = b * V + i;
        float pred = y_pred[offset];
        float target = y_true[offset];
        float val = 0.0f;
        if (!log_target) {
            val = target * (logf(fmaxf(target, eps)) - pred);
        } else {
            val = expf(target) * (target - pred);
        }
        sum += val;
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();

   
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        loss[b] = sdata[0];
    }
}

// -------------------------------------------------------------------
// Backward kernels
// -------------------------------------------------------------------

__global__ void kldiv_backward_kernel(const float* __restrict__ y_true,
                                      float* __restrict__ grad,
                                      int V,
                                      bool log_target) {
    int b = blockIdx.x;  // batch index
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        int offset = b * V + i;
        float target = y_true[offset];
        float res = (!log_target) ? -target : -expf(target);
        grad[offset] = res;
    }
}


__global__ void scale_kernel(float* data, int N, float factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        data[i] *= factor;
    }
}

// -------------------------------------------------------------------
// Host functions for forward and backward passes
// -------------------------------------------------------------------

// Forward pass host function.
void kldiv_forward(const float* y_pred, const float* y_true, float* loss,
                   int B, int V, float eps, bool log_target, Reduction reduction) {
    if (reduction == NONE) {
       
        int threads = V;
        dim3 grid(B);
        kldiv_forward_kernel_none<<<grid, threads>>>(y_pred, y_true, loss, V, eps, log_target);
    } else {
       
        int threads = 256;
        dim3 grid(B);
        size_t shared_mem_size = threads * sizeof(float);
        kldiv_forward_kernel_reduce<<<grid, threads, shared_mem_size>>>(y_pred, y_true, loss, V, eps, log_target);
    }
    cudaDeviceSynchronize();
}

// Backward pass host function.
void kldiv_backward(const float* y_true, float* grad,
                    int B, int V, bool log_target, float grad_output = 1.0f) {
    int threads = 256;
    dim3 grid(B);
    kldiv_backward_kernel<<<grid, threads>>>(y_true, grad, V, log_target);
    cudaDeviceSynchronize();

    // If grad_output is not 1, we scale the gradients!
    if (grad_output != 1.0f) {
        int total = B * V;
        int blockSize = 256;
        int numBlocks = (total + blockSize - 1) / blockSize;
        scale_kernel<<<numBlocks, blockSize>>>(grad, total, grad_output);
        cudaDeviceSynchronize();
    }
}


int main() {
    
    const int B = 2;
    const int V = 1024;
    size_t dataSize = B * V * sizeof(float);
    size_t lossSize = (B * ((BATCHMEAN == NONE) ? V : 1)) * sizeof(float);

   
    float* h_y_pred = new float[B * V];
    float* h_y_true = new float[B * V];
    float* h_loss   = new float[B]; 

    // Initialize
    for (int i = 0; i < B * V; i++) {
        h_y_pred[i] = 0.5f; 
        h_y_true[i] = 0.3f; 
    }

  
    float *d_y_pred, *d_y_true, *d_loss, *d_grad;
    cudaMalloc(&d_y_pred, dataSize);
    cudaMalloc(&d_y_true, dataSize);
 
    cudaMalloc(&d_loss, B * sizeof(float));
    cudaMalloc(&d_grad, dataSize);

  
    cudaMemcpy(d_y_pred, h_y_pred, dataSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_true, h_y_true, dataSize, cudaMemcpyHostToDevice);

    // forward pass.
    Reduction reduction = BATCHMEAN;
    kldiv_forward(d_y_pred, d_y_true, d_loss, B, V, 1e-10f, false, reduction);

   
    cudaMemcpy(h_loss, d_loss, B * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Forward loss (per batch):\n");
    for (int b = 0; b < B; b++) {
        printf("Batch %d: %f\n", b, h_loss[b]);
    }

    // backward pass.
    // This computes the gradient based on y_true.
    kldiv_backward(d_y_true, d_grad, B, V, false, 1.0f);

    
    float* h_grad = new float[B * V];
    cudaMemcpy(h_grad, d_grad, dataSize, cudaMemcpyDeviceToHost);
    printf("\nFirst 10 gradient values:\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", h_grad[i]);
    }
    printf("\n");


    cudaFree(d_y_pred);
    cudaFree(d_y_true);
    cudaFree(d_loss);
    cudaFree(d_grad);


    delete[] h_y_pred;
    delete[] h_y_true;
    delete[] h_loss;
    delete[] h_grad;

    return 0;
}
