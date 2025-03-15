import torch
import triton
import triton.language as tl

@triton.jit
def layer_norm_kernel(
    X, Y, W, B, M, V, N, EPS, BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x_ptrs = X + row_idx * N + cols
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    # Compute mean
    mean = tl.sum(x, axis=0) / N
    # Compute variance using multiplication instead of exponentiation
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / N
    inv_std = 1.0 / tl.sqrt(var + EPS)

    w_ptrs = W + cols
    b_ptrs = B + cols
    y_ptrs = Y + row_idx * N + cols

    w = tl.load(w_ptrs, mask=mask, other=1.0)
    b = tl.load(b_ptrs, mask=mask, other=0.0)

    y = diff * inv_std * w + b
    tl.store(y_ptrs, y, mask=mask)

def layer_norm(X, weight, bias, eps=1e-5):
    B, N = X.shape
    Y = torch.empty_like(X)

    grid = (B,)
    layer_norm_kernel[grid](
        X, Y, weight, bias, None, None, N, eps, BLOCK_SIZE=N
    )
    return Y

def main():
    B, N = 4, 128  # Batch size and feature dimension
    X = torch.randn(B, N, device='cuda', dtype=torch.float32)
    weight = torch.ones(N, device='cuda', dtype=torch.float32)
    bias = torch.zeros(N, device='cuda', dtype=torch.float32)

    Y = layer_norm(X, weight, bias)
    print("Output Shape:", Y.shape)
    print("Output:", Y)

if __name__ == "__main__":
    main()
