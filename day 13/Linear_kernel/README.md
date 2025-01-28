
## **Linear Kernel**

### Files Overview:
1. **`linear_kernel.cu`**  
   - The main program file orchestrates the Linear Kernel computation.  
   - Manages input/output data, memory allocations, and calls helper functions and CUDA kernels.  

2. **`helper_functions.h`**  
   - Declares utility functions for memory management, error handling, and matrix initialization.  
   - Contains declarations for CUDA memory allocation and status-checking functions.  

3. **`helper_functions.cpp`**  
   - Implements utility functions declared in `helper_functions.h`.  
   - Provides helper functions like `initializeRandomMatrix`, `allocateDeviceMemory`, and error-checking routines.  

4. **`cuda_kernels.h`**  
   - Declares the CUDA kernel for bias addition (`addBiasKernel`).  
   - Provides a function prototype for `performLinearLayerOperation` to manage matrix multiplication and bias addition.

5. **`cuda_kernels.cu`**  
   - Implements the CUDA kernel functions and their host-side wrappers.  
   - Contains:
     - **`addBiasKernel`**: Adds the bias to the output matrix in parallel using CUDA threads.  
     - **`performLinearLayerOperation`**: Combines matrix multiplication (using cuBLAS `cublasSgemm`) and bias addition.

---

## **Compiling the Linear Kernel**
Compile the Linear Kernel project using `nvcc`. Ensure you have CUDA and cuBLAS installed.  
```bash
nvcc linear_kernel.cu helper_functions.cpp cuda_kernels.cu -lcublas -o linear_layer
```

---

## **Execution Workflow**
1. **Initialization**:
   - Allocate and initialize input matrices (`host_input`, `host_weights`, `host_bias`) on the host (CPU).  
   - Transfer data to the device (GPU) for computation.  

2. **Matrix Multiplication**:
   - Perform `C = A × B` using cuBLAS.  
   - Input (`A`), weights (`B`), and output (`C`) dimensions are defined by `batch_size`, `input_features`, and `output_features`.

3. **Bias Addition**:
   - Use the `addBiasKernel` CUDA kernel to add bias to each output feature.

4. **Results**:
   - Transfer the results from the GPU back to the CPU for verification and further processing.  

5. **Clean-Up**:
   - Free both host and device memory, and destroy the cuBLAS handle.

---
