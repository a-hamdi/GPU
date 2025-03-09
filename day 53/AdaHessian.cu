#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

// AdaHessian update kernel
// Parameters:
//   theta:            current model parameters (to be updated)
//   grad:             gradient computed at theta
//   gradPerturbed:    gradient computed at theta + delta
//   m:                first moment estimate (accumulator for gradients)
//   v:                second moment estimate (accumulator for Hessian diag squared)
//   lr:               learning rate
//   beta1:            exponential decay rate for first moment
//   beta2:            exponential decay rate for second moment
//   epsilon:          small constant for numerical stability
//   delta:            finite difference perturbation value
//   N:                total number of parameters
__global__ void adaHessianUpdateKernel(
    float* theta,                
    const float* grad,           
    const float* gradPerturbed,  
    float* m,                    
    float* v,                    
    const float lr,              
    const float beta1,           
    const float beta2,           
    const float epsilon,         
    const float delta,           
    int N                        
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        // Approximate the Hessian diagonal using finite differences:
        float h_diag = (gradPerturbed[idx] - grad[idx]) / delta;
        
        // Update first moment (gradient) estimate:
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * grad[idx];
        
        // Update second moment (squared Hessian diag) estimate:
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * (h_diag * h_diag);
        
        // Update parameters using the AdaHessian rule:
        theta[idx] -= lr * m[idx] / (sqrtf(v[idx]) + epsilon);
    }
}

int main() {
    const int N = 10;             // Number of parameters for testing
    const int bytes = N * sizeof(float);
    const float lr = 0.01f;         // Learning rate
    const float beta1 = 0.9f;       // Decay rate for first moment
    const float beta2 = 0.999f;     // Decay rate for second moment
    const float epsilon = 1e-7f;    // Small constant for numerical stability
    const float delta = 1e-4f;      // Perturbation for finite differences

    // Host arrays
    float h_theta[N], h_grad[N], h_gradPerturbed[N], h_m[N], h_v[N];
    
    // Initialize arrays with dummy data
    for (int i = 0; i < N; i++) {
        h_theta[i] = 1.0f;          // Initial parameter value
        h_grad[i] = 0.1f;           // Dummy gradient
        // Simulate perturbed gradient: a small change from h_grad
        h_gradPerturbed[i] = 0.1f + 0.001f * i;
        h_m[i] = 0.0f;              // Initialize first moment
        h_v[i] = 0.0f;              // Initialize second moment
    }

    // Device arrays
    float *d_theta, *d_grad, *d_gradPerturbed, *d_m, *d_v;
    cudaMalloc((void**)&d_theta, bytes);
    cudaMalloc((void**)&d_grad, bytes);
    cudaMalloc((void**)&d_gradPerturbed, bytes);
    cudaMalloc((void**)&d_m, bytes);
    cudaMalloc((void**)&d_v, bytes);

    // Copy data from host to device
    cudaMemcpy(d_theta, h_theta, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad, h_grad, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gradPerturbed, h_gradPerturbed, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, h_m, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, bytes, cudaMemcpyHostToDevice);

    // Launch the kernel with an appropriate grid size
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    adaHessianUpdateKernel<<<gridSize, blockSize>>>(
        d_theta, d_grad, d_gradPerturbed, d_m, d_v,
        lr, beta1, beta2, epsilon, delta, N
    );

    // Wait for the GPU to finish
    cudaDeviceSynchronize();

    // Copy updated parameters and moment estimates back to host
    cudaMemcpy(h_theta, d_theta, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_m, d_m, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_v, d_v, bytes, cudaMemcpyDeviceToHost);

    // Print updated theta values
    printf("Updated theta values:\n");
    for (int i = 0; i < N; i++) {
        printf("%f ", h_theta[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_theta);
    cudaFree(d_grad);
    cudaFree(d_gradPerturbed);
    cudaFree(d_m);
    cudaFree(d_v);

    return 0;
}

