#include <iostream>
__global__ void MatrixAdd_C(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)  {
        for(int j=0;j<N;j++){

          C[i*N+j] = A[i*N+j] + B[i*N+j];

        }
    return;
    }
}



__global__ void MatrixAdd_B(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i >= N) && (j >= N)) { return ; }

    C[i*N+j] = A[i*N+j] + B[i*N+j];

    }

__global__ void MatrixAdd_D(const float* A, const float* B, float* C, int N) {
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < N)  {
        for(int i=0;i<N;i++){

          C[i*N+j] = A[i*N+j] + B[i*N+j];

        }

    }

}



int main() {
    const int N = 10;
    float *A, *B, *C;

    // initialize the input matrices
    A = (float *)malloc( N*N* sizeof(float));
    B = (float *)malloc(N*N* sizeof(float));
    C = (float *)malloc(N*N * sizeof(float));


    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i * N + j] = 1.0f;
            B[i * N + j] = 2.0f;
            C[i * N + j] = 0.0f;
        }
    }

    float *d_a, *d_b,*d_c;
    cudaMalloc((void **)&d_a,N*N*sizeof(float));
    cudaMalloc((void **)&d_b,N*N*sizeof(float));
    cudaMalloc((void **)&d_c,N*N*sizeof(float));
    cudaMemcpy(d_a,A,N*N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,B,N*N*sizeof(float),cudaMemcpyHostToDevice);

    dim3 dimBlock(32, 16);
    dim3 dimGrid(ceil(N / 32.0f), ceil(N/ 16.0f));
    MatrixAdd_B<<<dimGrid, dimBlock>>>(d_a, d_b, d_c,N);
    cudaDeviceSynchronize();

    cudaMemcpy(C,d_c,N*N*sizeof(float),cudaMemcpyDeviceToHost);
    printf("C:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {

            printf("%.2f ",C[i * N + j]); // Prints each element with 2 decimal precision
        }
        printf("\n"); // Adds a newline after each row
    }
     printf("A:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {

            printf("%.2f ", A[i * N + j]); // Prints each element with 2 decimal precision
        }
        printf("\n"); // Adds a newline after each row
    }
     printf("B:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {

            printf("%.2f ", B[i * N + j]); // Prints each element with 2 decimal precision
        }
        printf("\n"); // Adds a newline after each row
    }
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

}