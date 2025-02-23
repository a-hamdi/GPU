import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr, 
    M, N, K, 
    stride_am, stride_ak, 
    stride_bk, stride_bn, 
    stride_cm, stride_cn, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    
    # Compute the row and column index for the block
    row_idx = pid // (N // BLOCK_SIZE) * BLOCK_SIZE
    col_idx = pid % (N // BLOCK_SIZE) * BLOCK_SIZE
    
    # Create accumulators
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE):
        # Load submatrices of A and B into SRAM
        A = tl.load(A_ptr + (row_idx + tl.arange(0, BLOCK_SIZE))[:, None] * stride_am + (k + tl.arange(0, BLOCK_SIZE)) * stride_ak, mask=(row_idx + tl.arange(0, BLOCK_SIZE))[:, None] < M)
        B = tl.load(B_ptr + (k + tl.arange(0, BLOCK_SIZE))[:, None] * stride_bk + (col_idx + tl.arange(0, BLOCK_SIZE)) * stride_bn, mask=(col_idx + tl.arange(0, BLOCK_SIZE)) < N)

        # Matrix multiplication
        acc += tl.dot(A, B)
    
    # Store the result
    mask = (row_idx + tl.arange(0, BLOCK_SIZE))[:, None] < M and (col_idx + tl.arange(0, BLOCK_SIZE)) < N
    tl.store(C_ptr + (row_idx + tl.arange(0, BLOCK_SIZE))[:, None] * stride_cm + (col_idx + tl.arange(0, BLOCK_SIZE)) * stride_cn, acc, mask=mask)

def triton_matmul(A, B):
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    
    grid = (M // 16) * (N // 16)

    matmul_kernel[grid](
        A, B, C, 
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE=16
    )

    return C

# Example Usage
A = torch.randn(128, 128, device="cuda", dtype=torch.float32)
B = torch.randn(128, 128, device="cuda", dtype=torch.float32)

C = triton_matmul(A, B)
print(C)
