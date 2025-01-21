# CUDA 100 days Learning Journey

This document serves as a log of the progress and knowledge I gained while working on CUDA programming and studying the **PMPP (Parallel Programming and Optimization)** book.

---

## Day 1
### File: `vectadd.cu`
**Summary:**  
Implemented vector addition by writing a simple CUDA program. Explored how to launch a kernel to perform a parallelized addition of two arrays, where each thread computes the sum of a pair of values.  

**Learned:**  
- Basics of writing a CUDA kernel.
- Understanding of grid, block, and thread hierarchy in CUDA.  
- How to allocate and manage device (GPU) memory using `cudaMalloc`, `cudaMemcpy`, and `cudaFree`.  

### Reading:  
- Read **Chapter 1** of the PMPP book.  
  - Learned about the fundamentals of parallel programming, CUDA architecture, and the GPU execution model.

---

## Day 2
### File: `MatrixAdd.cu`
**Summary:**  
Worked on matrix addition using CUDA. Designed the grid and block layout to handle 2D matrices in parallel, with each element processed by an individual thread.  

**Learned:**  
- How to map 2D matrix data onto multiple threads.
- Understanding thread indexing in 2D grids and blocks using `threadIdx`, `blockIdx`, `blockDim`, and `gridDim`.  
- Synchronizing threads and avoiding race conditions when writing results to an output matrix.  

### Reading:  
- Read **Chapter 2** of the PMPP book.  
  - Learned about scalability of GPUs, massive parallelism, and how to configure problem data to match GPU thread hierarchies.  

---

## Day 3
### File: `Matrix_vec_mult.cu`
**Summary:**  
Implemented matrix-vector multiplication using CUDA. Each thread was set up to compute the dot product between a matrix row and the given vector. Optimized performance using shared memory.  

**Learned:**  
- How to perform dot products in parallel.
- Efficiently handling shared memory to avoid excessive global memory accesses and improve memory coalescing.
- Launching kernels for 1D or 2D thread configurations based on input data.  

### Reading:  
- Read **half of Chapter 3** of the PMPP book.  
 -Learned about Scalable Parallel Execution.

---

## Day 4
### File: `PartialSum.cu`
**Summary:**  
Worked on parallel reduction to compute the partial sum of an array. Implemented a tree-based reduction algorithm, minimizing warp divergence for better performance.  

**Learned:**  
- The concept of reduction in parallel programming.
- Techniques for minimizing warp divergence and balancing workload across threads.
- How to use shared memory effectively in reduction operations.  

### Reading:  
- Finished **Chapter 3** of the PMPP book.  
  - Learned about Scalable Parallel Execution including Resource Assignment and Thread Scheduling and Latency Tolerance

---

## Day 5
### File: `LayerNorm.cu`
**Summary:**  
Implemented Layer Normalization in CUDA, often used in deep learning models. Explored normalization techniques across batches and layers using reduction operations. Addressed the challenge of maintaining numerical stability during computation.  

**Learned:**  
- How to calculate mean and variance in parallel using reduction algorithms.
- Strategies to stabilize floating-point operations to prevent overflow or underflow issues.
- CUDA kernel optimization for workloads involving tensor computation.  

### Reading:  
- Read **Chapter 4** of the PMPP book.  
  -  Learned about memory optimizations and strategies for GPU performance tuning.

---

## Day 6
### File: `MatrixTranspose.cu`
**Summary:**  
Implemented CUDA-based matrix transposition. Optimized the implementation by leveraging shared memory to minimize global memory reads and writes. Ensured proper handling of edge cases when the matrix dimensions are not multiples of the block size.  

**Learned:**  
- How to optimize memory usage when working with global and shared memory.  
- Techniques to handle data alignment and padding for non-square matrices during transposition.  
- The importance of coalescing memory accesses in CUDA to improve performance.  

### Reading:  
- Read **Chapter 5** of the PMPP book.  
  - Learned about Performance Considerations including optimizing memory access patterns, advanced use of shared memory for performance and dynamic Partitioning of Resources .  
- Read **Chapter 6** of the PMPP book.  
  - Learned about Numerical Considerations including IEEE Format, Arithmetic Accuracy and Rounding and Linear Solvers and Numerical Stability. 
