import torch
import triton
import triton.language as tl
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
import numpy as np

# Triton kernel for a 2x2 Haar wavelet transform
@triton.jit
def wavelet_transform_kernel(
    input_ptr, output_LL_ptr, output_LH_ptr, output_HL_ptr, output_HH_ptr,
    H: tl.constexpr, W: tl.constexpr, stride: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    num_cols = W // BLOCK_SIZE  # Number of 2x2 blocks per row
    row = pid // num_cols
    col = pid % num_cols

    if (row * 2 < H) and (col * 2 < W):
        # Load a 2x2 block from the input image
        x00 = tl.load(input_ptr + (row * 2) * stride + (col * 2))
        x01 = tl.load(input_ptr + (row * 2) * stride + (col * 2 + 1))
        x10 = tl.load(input_ptr + (row * 2 + 1) * stride + (col * 2))
        x11 = tl.load(input_ptr + (row * 2 + 1) * stride + (col * 2 + 1))

        # Compute the Haar wavelet coefficients
        LL = (x00 + x01 + x10 + x11) / 4.0
        LH = (x00 - x01 + x10 - x11) / 4.0
        HL = (x00 + x01 - x10 - x11) / 4.0
        HH = (x00 - x01 - x10 + x11) / 4.0

        out_index = row * num_cols + col
        tl.store(output_LL_ptr + out_index, LL)
        tl.store(output_LH_ptr + out_index, LH)
        tl.store(output_HL_ptr + out_index, HL)
        tl.store(output_HH_ptr + out_index, HH)

def wavelet_transform(image_tensor):
    """
    Computes the Haar wavelet transform of a 2D image tensor.
    Assumes the image dimensions are even.
    """
    H, W = image_tensor.shape
    assert H % 2 == 0 and W % 2 == 0, "Image dimensions must be even."
    out_H, out_W = H // 2, W // 2

    # Create output tensors on the same device as input.
    LL = torch.empty((out_H, out_W), device=image_tensor.device, dtype=image_tensor.dtype)
    LH = torch.empty((out_H, out_W), device=image_tensor.device, dtype=image_tensor.dtype)
    HL = torch.empty((out_H, out_W), device=image_tensor.device, dtype=image_tensor.dtype)
    HH = torch.empty((out_H, out_W), device=image_tensor.device, dtype=image_tensor.dtype)

    # Total number of 2x2 blocks in the image
    num_blocks = out_H * out_W
    grid = (num_blocks,)  # 1D grid

    # Launch the Triton kernel
    wavelet_transform_kernel[grid](
        image_tensor, LL, LH, HL, HH,
        H, W, image_tensor.stride(0), BLOCK_SIZE=2
    )

    return LL, LH, HL, HH

def download_and_preprocess_image(url, size=(256, 256)):
    """
    Downloads an image from the given URL, converts it to grayscale,
    resizes it to the specified size, and returns a normalized torch tensor.
    """
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Error downloading image: HTTP {response.status_code}")
    
    try:
        img = Image.open(BytesIO(response.content)).convert("L")  # convert to grayscale
    except Exception as e:
        raise ValueError("Cannot identify image file. Check the URL or the content type.") from e
    
    img = img.resize(size)
    img_np = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1]
    return torch.from_numpy(img_np)

# Use an alternative cat image URL from cataas.com
cat_image_url = "https://cataas.com/cat"  # Returns a cat image

# Download and preprocess the cat image
cat_image_tensor = download_and_preprocess_image(cat_image_url)
# Move to CUDA for processing (ensure your environment has a CUDA-enabled GPU)
cat_image_tensor = cat_image_tensor.to("cuda")

# Compute the wavelet transform on the cat image
LL, LH, HL, HH = wavelet_transform(cat_image_tensor)

# Move the tensors to CPU for plotting
cat_image_cpu = cat_image_tensor.cpu().numpy()
LL_cpu = LL.cpu().numpy()
LH_cpu = LH.cpu().numpy()
HL_cpu = HL.cpu().numpy()
HH_cpu = HH.cpu().numpy()

# Plot the original cat image and the wavelet coefficients
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
axes[0].imshow(cat_image_cpu, cmap="gray")
axes[0].set_title("Original Cat Image")
axes[0].axis("off")

axes[1].imshow(LL_cpu, cmap="gray")
axes[1].set_title("LL (Approximation)")
axes[1].axis("off")

axes[2].imshow(LH_cpu, cmap="gray")
axes[2].set_title("LH (Horizontal)")
axes[2].axis("off")

axes[3].imshow(HL_cpu, cmap="gray")
axes[3].set_title("HL (Vertical)")
axes[3].axis("off")

axes[4].imshow(HH_cpu, cmap="gray")
axes[4].set_title("HH (Diagonal)")
axes[4].axis("off")

plt.tight_layout()
plt.show()
