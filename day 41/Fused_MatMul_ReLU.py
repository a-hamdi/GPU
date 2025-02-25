import torch
import triton
import triton.language as tl

@triton.jit
def matmul_relu_kernel(A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    row_idx = pid // (N // BLOCK_SIZE) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_idx = pid % (N // BLOCK_SIZE) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask_row = row_idx < M
    mask_col = col_idx < N

    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE):
        a = tl.load(A + row_idx[:, None] * stride_am + (k + tl.arange(0, BLOCK_SIZE)) * stride_ak, mask=mask_row[:, None])
        b = tl.load(B + (k + tl.arange(0, BLOCK_SIZE))[:, None] * stride_bk + col_idx[None, :] * stride_bn, mask=mask_col[None, :])
        acc += tl.dot(a, b)
    
    acc = tl.maximum(acc, 0)  # Apply ReLU activation
    tl.store(C + row_idx[:, None] * stride_cm + col_idx[None, :] * stride_cn, acc, mask=mask_row[:, None] & mask_col[None, :])


def matmul_relu(A: torch.Tensor, B: torch.Tensor):
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    BLOCK_SIZE = 16
    
    grid = (M // BLOCK_SIZE) * (N // BLOCK_SIZE)
    
    matmul_relu_kernel[grid](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )
    return C

# Main function to test the kernel
def main():
    torch.manual_seed(0)
    M, N, K = 64, 64, 64  # Matrix dimensions
    A = torch.randn((M, K), device='cuda', dtype=torch.float32)
    B = torch.randn((K, N), device='cuda', dtype=torch.float32)
    
    C = matmul_relu(A, B)
    print("Result:", C)

if __name__ == "__main__":
    main()
