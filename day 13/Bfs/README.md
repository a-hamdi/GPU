
### File: `bfs.h`
**Summary:**  
Created a header file for the Parallelized Breadth-First Search (BFS) project. This file includes essential macros, function declarations, and utility functions to facilitate the implementation of the BFS algorithm on the GPU.  

---

### File: `bfs_kernel.cu`
**Summary:**  
Implemented the CUDA kernel for parallel BFS in the `bfs_kernel.cu` file. The kernel utilizes atomic operations to ensure thread-safe updates to labels, enabling multiple threads to explore graph edges concurrently.  

**Learned:**  
- How to design a CUDA kernel for graph traversal using parallelization strategies.
- Used `atomicCAS` to manage concurrent access to shared label data, preventing data races.
- The concept of marking nodes at each BFS level and how to signal completion of traversal via atomic flags.

---

### File: `bfs_gpu.cu`
**Summary:**  
Developed the main GPU function `bfs_gpu` that integrates the graph traversal functionality, orchestrating the initialization of kernel launches and memory management for the BFS operation.  

**Learned:**  
- How to manage kernel launches based on the graph size and structure.
- Strategies for handling the BFS frontier and updating labels.

---

### File: `bfs_cpu.c`
**Summary:**  
Completed implementations for the CPU BFS version and a random graph generator. This aids in testing the correctness of the GPU BFS by providing a reliable CPU counterpart for comparison.  