import torch
import triton
import triton.language as tl

@triton.jit
def conv1d_kernel(
    input_ptr, kernel_ptr, output_ptr, 
    BATCH, IN_CH, OUT_CH, IN_LEN, KERNEL_SIZE, STRIDE, OUT_LEN, 
    BLOCK_B: tl.constexpr, BLOCK_OUT: tl.constexpr
):
    b = tl.program_id(0)
    oc = tl.program_id(1)
    out_idx = tl.program_id(2) * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    
    mask = out_idx < OUT_LEN
    
    acc = tl.zeros([BLOCK_OUT], dtype=tl.float32)
    
    for ic in range(IN_CH):
        for k in range(KERNEL_SIZE):
            in_idx = out_idx * STRIDE + k
            in_mask = in_idx < IN_LEN
            
            input_offset = ((b * IN_CH + ic) * IN_LEN) + in_idx
            kernel_offset = ((oc * IN_CH + ic) * KERNEL_SIZE) + k
            
            input_val = tl.load(input_ptr + input_offset, mask=in_mask, other=0.0)
            kernel_val = tl.load(kernel_ptr + kernel_offset)
            
            acc += input_val * kernel_val
    
    output_offset = (b * OUT_CH + oc) * OUT_LEN + out_idx
    tl.store(output_ptr + output_offset, acc, mask=mask)


def conv1d_triton(input, kernel, stride=1):
    BATCH, IN_CH, IN_LEN = input.shape
    OUT_CH, _, KERNEL_SIZE = kernel.shape
    OUT_LEN = (IN_LEN - KERNEL_SIZE) // stride + 1
    
    output = torch.empty((BATCH, OUT_CH, OUT_LEN), device=input.device, dtype=torch.float32)
    
    grid = (BATCH, OUT_CH, (OUT_LEN + 31) // 32)
    
    conv1d_kernel[grid](
        input, kernel, output, 
        BATCH, IN_CH, OUT_CH, IN_LEN, KERNEL_SIZE, stride, OUT_LEN,
        BLOCK_B=1, BLOCK_OUT=32
    )
    
    return output

# Example usage
BATCH, IN_CH, IN_LEN = 1, 3, 64
OUT_CH, KERNEL_SIZE, STRIDE = 8, 5, 1

torch.manual_seed(0)
input_tensor = torch.randn(BATCH, IN_CH, IN_LEN, device='cuda')
kernel_tensor = torch.randn(OUT_CH, IN_CH, KERNEL_SIZE, device='cuda')

output_tensor = conv1d_triton(input_tensor, kernel_tensor, STRIDE)
print(output_tensor.shape)
