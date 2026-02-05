"""
Individual Kernel Effect Comparisons

Generates side-by-side input/output images for each kernel type,
showing the effect of convolution clearly on the Brandenburg Gate photograph.

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
    img = Image.open(IMAGE_PATH).convert('L')

    # Resize for faster processing while maintaining aspect ratio
    img = img.resize((400, 267))

    # Convert to numpy array with float64 for precision during convolution
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
# Step 2: Define kernels with their properties
# =============================================================================
# Each kernel has a specific mathematical structure that determines its effect

kernels = {
    'identity': {
        # Identity kernel: Only the center pixel contributes (weight = 1)
        # All other weights are zero, so output equals input
        'kernel': np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=np.float64),
        'caption': 'Identity kernel: output matches input'
    },
    'blur': {
        # Box blur: All 9 pixels contribute equally (each weight = 1/9)
        # The output is the average of the 3x3 neighborhood
        'kernel': np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ], dtype=np.float64) / 9.0,
        'caption': 'Blur kernel: edges become soft gradients'
    },
    'sharpen': {
        # Sharpen: Center weight (5) minus neighbors (-1 each)
        # Sum of weights = 1, so brightness is preserved
        # Enhances differences between center and neighbors
        'kernel': np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float64),
        'caption': 'Sharpen kernel: edges and details enhanced'
    },
    'edge_detect': {
        # Laplacian edge detection: Center (8) minus all neighbors (-1 each)
        # Sum of weights = 0, so uniform regions become black
        # Only areas where pixel values change produce non-zero output
        'kernel': np.array([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ], dtype=np.float64),
        'caption': 'Edge detection kernel: only boundaries visible'
    }
}

# =============================================================================
# Step 3: Convolution function
# =============================================================================
def apply_convolution(image, kernel):
    """
    Apply a convolution kernel to a grayscale image.

    Parameters:
        image (np.ndarray): 2D array of pixel values
        kernel (np.ndarray): 2D array of kernel weights

    Returns:
        np.ndarray: Convolved image (same size as input)
    """
    kernel_size = kernel.shape[0]
    pad = kernel_size // 2

    # Create output array
    output = np.zeros_like(image)

    # Pad the image to handle border pixels
    # 'edge' mode extends by replicating edge pixels
    padded = np.pad(image, pad, mode='edge')

    height, width = image.shape

    # Core convolution loop: slide kernel over every pixel
    for y in range(height):
        for x in range(width):
            # Extract region under the kernel
            region = padded[y:y + kernel_size, x:x + kernel_size]

            # Compute weighted sum (convolution operation)
            output[y, x] = np.sum(region * kernel)

    return output

# =============================================================================
# Step 4: Generate individual comparison images
# =============================================================================
for name, data in kernels.items():
    kernel = data['kernel']
    caption = data['caption']

    # Apply convolution with this kernel
    result = apply_convolution(canvas, kernel)

    # Clip to valid display range [0, 255]
    result = np.clip(result, 0, 255)

    # Create side-by-side comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)

    # Left panel: Original input image
    axes[0].imshow(canvas, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Input', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Right panel: Convolved output
    axes[1].imshow(result, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('Output', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    # Add caption describing the effect
    fig.suptitle(caption, fontsize=11, y=0.02)

    plt.tight_layout()
    plt.savefig(f'kernel_{name}.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"Generated kernel_{name}.png")

print("\nAll individual kernel effect images generated successfully.")
