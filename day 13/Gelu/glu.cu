#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>

__global__ void gelu_kernel(float* data, int size) {

  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<size){
    data[i]=0.5*data[i]*(1.0+erff(data[i]/sqrt(2.0)));
  }

}





int main(){
    const int N=1000000;
    float A[N];
    for (int i = -2; i < N; i++) {
        A[i] = -1*(float)i/2;
    }
  
        for (int i = 0; i < 10; ++i) {
        std::cout << "A[" << i << "]: " << A[i] << std::endl;
    }

    float *d_A;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x);
    gelu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, N);
    cudaDeviceSynchronize();
    cudaMemcpy(A, d_A, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    for (int i =0 ; i < 10; ++i) {
        std::cout << "A[" << i << "]: " << A[i] << std::endl;
    }

         




}