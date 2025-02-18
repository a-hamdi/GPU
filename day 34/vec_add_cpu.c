#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

#define UNROLL_FACTOR 4  // Loop unrolling factor

void vector_add(double *A, double *B, double *C, int size) {
    int i, limit = size - (size % UNROLL_FACTOR);
    for (i = 0; i < limit; i += UNROLL_FACTOR) {
        C[i]   = A[i]   + B[i];
        C[i+1] = A[i+1] + B[i+1];
        C[i+2] = A[i+2] + B[i+2];
        C[i+3] = A[i+3] + B[i+3];
    }
    for (; i < size; i++) {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char **argv) {
    int rank, size, N = atoi(argv[1]);  // Take vector size as argument
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int chunk_size = N / size;
    int start = rank * chunk_size;
    int end = (rank == size - 1) ? N : start + chunk_size;

    double *A = (double*) malloc(chunk_size * sizeof(double));
    double *B = (double*) malloc(chunk_size * sizeof(double));
    double *C = (double*) malloc(chunk_size * sizeof(double));

    for (int i = 0; i < chunk_size; i++) {
        A[i] = i + rank;
        B[i] = i - rank;
    }

    double t1 = MPI_Wtime();
    vector_add(A, B, C, chunk_size);
    double t2 = MPI_Wtime();

    if (rank == 0) {
        printf("CPU (MPI + Unrolling): %lf miliseconds\n", (t2 - t1)*1000.0);
    }

    free(A); free(B); free(C);
    MPI_Finalize();
    return 0;
}
