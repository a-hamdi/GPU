# CUDA 100 days Learning Journey

This document serves as a log of the progress and knowledge I gained while working on CUDA programming and studying the **PMPP (Parallel Programming and Optimization)** book.

Mentor: https://github.com/hkproj/


Bro in the 100 days challenge: https://github.com/1y33/100Days


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

## Day 7

### File: `one_d_convolution.cu`
**Summary:**  
Implemented a simple 1D convolution algorithm using CUDA. This involved sliding a kernel (or filter) over an input array and computing the weighted sum of elements. Each thread was assigned to compute the convolution at a specific position in the output array.  

**Learned:**  
- Basics of 1D convolution in parallel, including mapping threads to positions in the output array.
- How to handle boundary conditions (halo cells) when the kernel partially overlaps the input array bounds.
- Importance of memory layout and contiguous access for kernel weights and input arrays to maximize performance.

---

### File: `one_d_convolution_with_tiling.cu`
**Summary:**  
Implemented an optimized version of the 1D convolution algorithm using tiling and shared memory. Divided the input array into tiles and loaded data into shared memory, minimizing global memory accesses for better performance. Used halo cells to handle edge cases where kernel overlap extended into neighboring tiles.  

**Learned:**  
- Tiling in CUDA: Dividing input data into manageable chunks and leveraging shared memory to reduce global memory latency.
- Use of **halo cells** to ensure correctness at tile boundaries during convolution.
- How to balance computation and memory usage in tiled algorithms to improve performance.
- Proper synchronization of threads within a block (using `__syncthreads()`) to ensure data consistency in shared memory.

---
### File: `2d_convolution_with_tiling.cu`  
**Summary:**  
Implemented a 2D convolution algorithm with tiling optimization using CUDA. Divided the input matrix into tiles and leveraged shared memory to minimize global memory accesses, ensuring efficient computation of the convolution kernel across the matrix. Handled boundary conditions using halo cells to process edges and corners correctly.  

**Learned:**  
- Extended tiling techniques from 1D to 2D data structures for efficient parallel computation.  
- Optimized global memory access by using shared memory for each tile.  
- Synchronization of threads for consistent shared memory usage within a block (`__syncthreads()` for proper execution order).  
- Efficient handling of edge cases and boundary cells in 2D convolution.  

--- 

### Reading:  
- Read **Chapter 7** of the PMPP book.  
  - Learned about parallel patterns for convolution, including basic algorithms, memory optimizations with constant and shared memory, and tiling techniques with halo cells for 1D and 2D convolution.

## Day 8  
### File: `prefixsum_brent_kung_algorithm.cu`  
**Summary:**  
Implemented the Brent-Kung algorithm for parallel prefix sum (scan) in CUDA, designing a work-efficient strategy to compute prefix sums across an array.  

**Learned:**  
- The fundamentals of hierarchical parallel scan algorithms and the Brent-Kung approach for work efficiency.
- How to divide the scan operation into an **up-sweep (reduce)** phase and a **down-sweep** phase using shared memory for efficient computation.  
- Optimized thread synchronization and memory usage for large input arrays.  

### Reading:  
- Read **Chapter 8** of the PMPP book.  
  - Learned about different parallel patterns for prefix sum computation, focusing on performance, memory access efficiency, and work-efficient algorithms like hierarchical scans.  
- Read **Chapter 9** of the PMPP book.  
  - Learned about different parallel patterns for Parallel Histogram Computation, focusing on Atomic Operations,  Interleaved Partitioning, Privatization and Aggregation.  

### Day 9  

### File: `flash_attention_forward.cu`  
**Summary:**  
Implemented a forward pass for Flash Attention in CUDA, based on the Flash Attention paper. The code is still a work in progress and might produce incorrect results. A refined and fully functional version will be updated in the coming days.  

**Learned:**  
- Explored the fundamentals of Flash Attention, including its memory-efficient mechanism for attention computation.  
- Gained insights into optimizing CUDA kernels for operations like softmax and scaling factors used in attention.  
- Identified potential challenges in achieving numerical stability and correctness when implementing complex attention mechanisms.  

### Reading:  
- Read the **Flash Attention paper**.  
  - Learned about the key concepts of reducing memory overhead in attention computation, streamlining the matrix multiplication process, and ensuring efficient scaling for large models.


### Day 10:
### File: `flash_attention_forward.cu` 
Optimized and corrected yesterday's forward pass for Flash Attention in CUDA, based on the Flash Attention paper. The code is still a work in progress!

### File: `torch_test.py` 
Torch code to check the results of flash_attention_forward kernel.

### Blog: `Understanding Flash Attention (Forward) with CUDA`
A blog on flash attention (forward algorithm) explaining the parts of my code. I'll try to make it more intuitive with drawings as soon as I have time.

### Day 11
### File: `sparse_MatrixVecMult_Hybrid_ELL_COO.cu`
**Summary:**  
Completed the implementation of a highly optimized sparse matrix-vector multiplication (SpMV) algorithm using a hybrid approach that combines ELL (Ellpack) and COO (Coordinate) formats. This implementation focuses on minimizing memory overhead while maximizing computational efficiency across the sparsity of the input matrix.

**Learned:**  
- Explored the principles and benefits of different sparse matrix representations, namely ELL and COO formats.
- Implemented hybrid techniques to optimize performance by balancing memory access patterns and ensuring efficient data locality.
- Benchmarked the performance of the CUDA implementation against PyTorch to evaluate the efficiency and correctness of the optimized SpMV algorithm.

### Reading:  
- Completed **Chapter 10** of the PMPP book.  
  - Gained insights into parallel patterns for sparse matrix computations, focusing on the background of sparse data handling, parallel SpMV using CSR formats, and padding and transposition techniques for optimization.  
  - Learned about utilizing hybrid approaches to manage padding effectively and methods for sorting and partitioning to enhance regularization in sparse data.

### File: `benchmark.py`
**Summary:**  
Developed a benchmarking script to evaluate the performance of the custom CUDA SpMV implementation against PyTorch's built-in functions. This benchmark facilitates comparative analysis of execution times and ensures that the implementation meets expected performance standards.

### Blog:  
- Wrote a blog post titled **"Learning CUDA with a Weak GPU or No GPU at All: Yes, You Can!"**  
  - Addressed common misconceptions regarding GPU programming and provided practical tips for learners with limited hardware resources. The blog offers insights on optimizing CPU-based implementations and highlights methods to learn CUDA fundamentals without direct access to a powerful GPU.

**Link to Blog:**  
[Learning CUDA with a Weak GPU or No GPU at All: Yes, You Can!](https://hamdi.bearblog.dev/learning-cuda-with-a-weak-gpu-or-no-gpu-at-all-yes-you-can/)

## Day 12
### File: `merge_sort.cu`
**Summary:**  
Implemented the Merge Sort algorithm using CUDA. The implementation focuses on merging two sorted arrays into a single sorted array using a parallel approach. The kernel utilizes a co-rank function to find positions in the combined array for inserting elements from the two sorted input arrays efficiently.  

**Learned:**  
- Explored the fundamentals of merge sort and its parallelization strategies.
- Implemented the co-rank function which assists in finding the correct position of elements while merging two sorted arrays.
- Developed a parallel merge kernel that utilizes the GPU's capabilities for concurrent execution, enhancing performance beyond a sequential merge approach.


### Reading:
- Read **Chapter 11** of the PMPP book.  
  - Covered various aspects of merge sort parallel pattern. Key sections included:
    - **Background**: Understanding the merge sort algorithm and its significance in parallel processing.
    - **Sequential Merge Algorithm**: Key insights into how merge operations are typically conducted sequentially.
    - **Parallelization Approach**: Strategies for achieving parallelism in merge sort, highlighting the expected performance benefits.
    - **Co-Rank Function Implementation**: Understanding how the co-rank function is used to determine merging positions effectively.
    - **Basic and Tiled Merge Kernel**: Learning about different kernel designs including basic parallel merge kernels and more advanced tiled merge techniques for optimizing data access patterns.


## Day 13
I coded a Breadth first search optimized kernel, check this for more details: [BFS](./day%2013/Bfs/README.md) .

I also coded  Gelu activation kernel, check this for more details: [Gelu](./day%2013/Gelu/README.md) .

And also coded a full linear layer that treats batches using cublas: [Linear_kernel](./day%2013/Glu/README.md) .

---

### Reading:
- Read **Chapter 12** of the PMPP book.  
  - Explored parallel patterns for graph searches, covering:
    - Background on graph structures and traversal mechanisms.
    - Detailed sections on implementing both sequential and parallel BFS functions.
    - Insights into optimizing graph traversal performance, including memory bandwidth considerations and load balancing strategies in parallel algorithms.
- Read **Chapter 13** of the PMPP book.
  - Learned about the fundamentals of CUDA Dynamic Parallelism, including:
    - The basics and overview of dynamic parallelism in CUDA.
    - How memory visibility works, especially in the context of different memory types (global, shared, local).
    - Memory management strategies and the impact of nesting depth on kernel launches.
    - Synchronization techniques, streams, and events for managing concurrent operations within dynamic kernels.
    - Studied a more complex example about Bezier curve calculations both with and without dynamic parallelism, enhancing my understanding of recursive  
---

### Future Plans:
- Optimize the BFS implementation using hierarchical queues for better memory usage and performance.
- Explore additional enhancements and optimizations discussed in Chapter 12 to refine the BFS algorithm further.
- Prepare a performance comparison between CPU and GPU implementations in the subsequent days.

## Day 14

### File: `cmpFHD.cu`
**Summary:**  
Implemented the FHD (Fully-Hybrid Domain) algorithm for non-Cartesian magnetic resonance imaging (MRI) reconstruction in CUDA. The code focuses on optimizing the parallelism structure to handle iterative reconstruction efficiently, aiming to balance computational load while reducing memory footprint.

**Learned:**  
- Gained insights into non-Cartesian MRI imaging techniques and their relevance in modern medical imaging applications.
- Developed an understanding of iterative reconstruction methods and how parallelization can significantly improve performance in reconstructing images from non-Cartesian data.
- Implemented optimizations to address common challenges in MRI reconstruction, such as memory bandwidth limitations and computational heavy-lifting.

### File: `cmpFHD_real_image.cu`
**Summary:**  
Built upon the previous implementation of the FHD algorithm to include real image reading and processing capabilities. This version takes an actual image, applies the FHD reconstruction algorithm, and outputs the reconstructed image, demonstrating practical applicability of the CUDA code.

**Learned:**  
- Expanded the previous understanding of memory management and kernel optimization by integrating real-world data processing into the workflow.
- Familiarized myself with image I/O operations in CUDA, allowing for the handling of real data as input for reconstruction algorithms.


### Reading:
- Completed **Chapter 14** of the PMPP book.  
  - Delved into the case study of non-Cartesian magnetic resonance imaging, which provided:
    - Background on the principles and necessities driving advancements in MRI technology.
    - A comprehensive look at iterative reconstruction techniques that enhance image quality using statistical estimation methods.
    - Detailed steps on optimizing the kernel parallelism structure to maximize performance and minimize memory constraints in handling MRI data.
    - Insights into experimental performance tuning, particularly the advantages of leveraging hardware trigonometry functions to achieve rapid computations.

---


### Day 15

#### File: `flash_attention_backprop.cu`
**Summary:**  
Implemented the backpropagation for Flash Attention in CUDA, continuing from the forward pass developed earlier. The backpropagation step computes the gradients required for training the attention mechanism. However, a small issue arose where some of the gradients are outputting as zero at certain points, which will be addressed and fixed in the coming days.

**Learned:**  
- Explored the process of backpropagation in the context of Flash Attention, including the calculation of gradients for the attention weights and input matrices.
- Worked on integrating gradient calculation with memory optimization techniques to maintain efficiency, consistent with the original forward pass.
- Identified potential issues related to numerical stability when dealing with gradient flow in CUDA, specifically in the attention layer.

---

#### File: `cnn.cu`
**Summary:**  
Developed a Convolutional Neural Network (CNN) implementation in CUDA, including both forward and backward passes with pooling layers. Used the unrolling trick for improved performance in the backward pass, optimizing the matrix operations involved.

**Learned:**  
- Implemented the core components of a CNN in CUDA, including convolutions, activations, pooling layers, and backpropagation.
- Utilized the unrolling trick to optimize it, improving the performance of matrix multiplications and gradient calculations.
- Gained deeper understanding of the computational requirements for CNN training on GPUs and the importance of efficient memory access patterns and parallelism in deep learning.

---

### Reading:  
- **Chapter 15:** *Application Case Study—Molecular Visualization and Analysis*  
  - Delved into the background and practical aspects of molecular visualization in parallel computing.  
  - Learned about the importance of thread granularity adjustments and memory coalescing in visualizing large-scale molecular structures using CUDA.

- **Chapter 16:** *Application Case Study—Machine Learning*  
  - Focused on Convolutional Neural Networks (ConvNets) and their implementation in CUDA.  
  - Covered key concepts such as basic layers, backpropagation, and the reduction of convolutional layers to matrix multiplication for optimization.  
  - Explored the cuDNN library and its use in accelerating deep learning operations.

- **Chapter 17:** *Parallel Programming and Computational Thinking*  
  - Studied the core principles of parallel computing, including problem decomposition, algorithm selection, and computational thinking.  
  - Focused on strategies for optimizing memory locality and shared memory usage in parallel applications.

---



### Future challenges:
- Day 15 - mandatory FA2-forward
- Day 20 - mandatory FA2-bakcwards
- Day 20 - optional fused chunked CE loss + backwards. we can use Liger Kernel as reference implementation to copy. 