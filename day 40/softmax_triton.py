import triton
import triton.language as tl
import torch

@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, stride_n, stride_f, N, F, BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)  # Each row is processed by one program instance
    offsets = tl.arange(0, BLOCK_SIZE)  # Column offsets
    mask = offsets < F  # Mask to avoid out-of-bounds access

    # Load input row into shared memory
    row_start = input_ptr + row_idx * stride_n
    values = tl.load(row_start + offsets, mask=mask, other=-float("inf"))

    # Compute max for numerical stability
    row_max = tl.max(values, axis=0)
    values = values - row_max

    # Compute exp and sum
    exp_values = tl.exp(values)
    row_sum = tl.sum(exp_values, axis=0)

    # Normalize
    softmax_vals = exp_values / row_sum

    # Store result
    row_output_start = output_ptr + row_idx * stride_n
    tl.store(row_output_start + offsets, softmax_vals, mask=mask)

def softmax_triton(x):
    N, F = x.shape
    output = torch.empty_like(x)
    
    grid = (N,)
    BLOCK_SIZE = triton.next_power_of_2(F)
    
    softmax_kernel[grid](
        output,
        x,
        x.stride(0),
        x.stride(1),
        N,
        F,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

# Example usage
x = torch.randn(4, 128, device="cuda")  # Batch of 4 vectors of size 128
y = softmax_triton(x)
print(y)
