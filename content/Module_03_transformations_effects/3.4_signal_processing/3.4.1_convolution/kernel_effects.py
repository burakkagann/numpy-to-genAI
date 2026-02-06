"""
Kernel Effects Comparison

Demonstrates how different convolution kernels produce different effects
on a real photograph (Brandenburg Gate):
- Identity: No change (passes through original)
- Blur: Smooths edges by averaging neighbors
- Sharpen: Enhances edges by emphasizing center pixel
- Edge Detection: Highlights boundaries between light and dark

Implementation inspired by:
- SciPy ndimage.convolve documentation
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.convolve.html
- OpenCV filter2D tutorial
  https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
- Gonzalez, R.C. and Woods, R.E. (2018). Digital Image Processing, 4th ed.
  Chapter 3: Intensity Transformations and Spatial Filtering
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# =============================================================================
# Step 1: Load the Brandenburg Gate image
# =============================================================================
# Path to the shared Brandenburg Gate image (relative from this directory)
IMAGE_PATH = '../../../../../_static/images/bbtor.jpg'

# Check if the image file exists
if os.path.exists(IMAGE_PATH):
    # Load image and convert to grayscale for convolution
    # Grayscale simplifies processing: one channel instead of three (RGB)
    img = Image.open(IMAGE_PATH).convert('L')

    # Resize for faster processing while maintaining aspect ratio
    # Original is 1920x1280, we scale down to ~400px width
    img = img.resize((400, 267))

    # Convert to numpy array with float64 for precision during convolution
    # Float64 prevents integer overflow when kernel values are multiplied
    canvas = np.array(img, dtype=np.float64)
    print(f"Loaded Brandenburg Gate image: {canvas.shape[1]}x{canvas.shape[0]} pixels")
else:
    # Fallback: create a synthetic test image if Brandenburg not found
    print(f"Warning: Image not found at {IMAGE_PATH}")
    print("Creating synthetic test pattern as fallback...")

    SIZE = 267
    canvas = np.zeros((SIZE, 400), dtype=np.float64)

    # Add a white rectangle
    canvas[40:80, 30:90] = 255

    # Add a white circle
    center_y, center_x = 120, 60
    for y in range(SIZE):
        for x in range(400):
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if distance < 30:
                canvas[y, x] = 255

    # Add a gradient region
    canvas[40:160, 120:180] = np.tile(np.linspace(0, 255, 60), (120, 1))

# =============================================================================
# Step 2: Define different kernels
# =============================================================================
# Each kernel produces a different effect based on its weight distribution.
# The kernel slides over the image, computing weighted sums at each position.

kernels = {
    'Original': None,  # Special case - no convolution applied

    # Blur kernel: All weights equal, averages the 3x3 neighborhood
    # Dividing by 9 normalizes the kernel so brightness is preserved
    'Blur (Box)': np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ], dtype=np.float64) / 9.0,

    # Sharpen kernel: Center weight > 1, negative neighbors
    # This amplifies the center pixel relative to its surroundings
    'Sharpen': np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float64),

    # Edge detection (Laplacian): Positive center, negative neighbors
    # Uniform regions become zero; edges (where values change) become bright
    'Edge Detect': np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ], dtype=np.float64)
}

# =============================================================================
# Step 3: Apply convolution function
# =============================================================================
def apply_convolution(image, kernel):
    """
    Apply a convolution kernel to a grayscale image.

    The convolution operation slides the kernel over every pixel position,
    computes element-wise multiplication with the underlying region,
    and sums the result to produce the output pixel value.

    Parameters:
        image (np.ndarray): 2D array of pixel values (grayscale, float64)
        kernel (np.ndarray): 2D array of kernel weights (e.g., 3x3)

    Returns:
        np.ndarray: Convolved image (same size as input)
    """
    if kernel is None:
        return image.copy()

    kernel_size = kernel.shape[0]
    pad = kernel_size // 2

    # Create output array with same dimensions as input
    output = np.zeros_like(image)

    # Pad the image to handle border pixels
    # 'edge' mode repeats the outermost pixel values outward
    # This avoids black borders in the output
    padded = np.pad(image, pad, mode='edge')

    height, width = image.shape

    # Slide the kernel over every pixel position
    for y in range(height):
        for x in range(width):
            # Extract the region that the kernel covers
            region = padded[y:y + kernel_size, x:x + kernel_size]

            # Element-wise multiplication and sum: this IS convolution
            output[y, x] = np.sum(region * kernel)

    return output

# =============================================================================
# Step 4: Apply each kernel and create comparison grid
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=150)
axes = axes.flatten()

for idx, (name, kernel) in enumerate(kernels.items()):
    # Apply convolution with current kernel
    result = apply_convolution(canvas, kernel)

    # Clip values to valid display range [0, 255]
    # Edge detection can produce negative values and values > 255
    result = np.clip(result, 0, 255)

    # Display the result
    axes[idx].imshow(result, cmap='gray', vmin=0, vmax=255)
    axes[idx].set_title(name, fontsize=14, fontweight='bold', pad=10)
    axes[idx].axis('off')

    # Add kernel visualization overlay for non-original images
    if kernel is not None:
        # Format kernel values as a small text block
        kernel_text = '\n'.join([' '.join([f'{v:5.1f}' for v in row])
                                  for row in kernel])
        axes[idx].text(0.02, 0.02, f'Kernel:\n{kernel_text}',
                      transform=axes[idx].transAxes,
                      fontsize=7, fontfamily='monospace',
                      verticalalignment='bottom',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.suptitle('Different Kernels Produce Different Effects',
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('kernel_effects.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("Kernel effects comparison saved as kernel_effects.png")
print()
print("Effects demonstrated:")
print("  - Original: The input image (no convolution)")
print("  - Blur: Smooths edges by averaging neighbors")
print("  - Sharpen: Enhances edges and details")
print("  - Edge Detect: Highlights boundaries (edges appear white)")
