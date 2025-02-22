import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(
    x_ptr, y_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def vector_add(x: torch.Tensor, y: torch.Tensor):
    assert x.shape == y.shape, "Input tensors must have the same shape"
    
    N = x.numel()
    output = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    vector_add_kernel[grid](
        x_ptr=x, y_ptr=y, output_ptr=output, N=N, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

# Example usage:
x = torch.randn(10000, device='cuda')
y = torch.randn(10000, device='cuda')
output = vector_add(x, y)
print(output)