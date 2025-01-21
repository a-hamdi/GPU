#include <iostream>

__global__ void vectorMatrixMult(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
      float sum=0.0f;
      for (int j = 0; j < N; j++) {
         sum += A[i*N+j]*B[j];
      }
      C[i]=sum;
}}

int main() {
    //initialize the matrix
    const int N = 10;
    float *A, *B, *C;

    // initialize the input matrices
    A = (float *)malloc( N*N* sizeof(float));
    B = (float *)malloc(N*sizeof(float));
    C = (float *)malloc(N*sizeof(float));


    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i * N + j] = 1.0f;
        }
        B[i] = 2.0f;
        C[i] = 0.0f;
    }

    float *d_a, *d_b,*d_c;
    cudaMalloc(&d_a,N*N*sizeof(float));
    cudaMalloc(&d_b,N*sizeof(float));
    cudaMalloc(&d_c,N*sizeof(float));
    cudaMemcpy(d_a,A,N*N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,B,N*sizeof(float),cudaMemcpyHostToDevice);
    int blocksize=256;
    int gridsize = (N + blocksize - 1) / blocksize;
    vectorMatrixMult<<<gridsize,blocksize>>>(d_a,d_b,d_c,N);

  cudaDeviceSynchronize();
cudaMemcpy(C,d_c,N*sizeof(float),cudaMemcpyDeviceToHost);

     printf("A:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {

            printf("%.2f ", A[i * N + j]); // Prints each element with 2 decimal precision
        }
        printf("\n"); // Adds a newline after each row
    }

    printf("C:\n");
    for (int i = 0; i < N; i++) {


            printf("%.2f ",C[i]); // Prints each element with 2 decimal precision

    }
printf("\n");
     printf("B:\n");
    for (int i = 0; i < N; i++) {


            printf("%.2f ", B[i ]); // Prints each element with 2 decimal precision

    }



    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

}
