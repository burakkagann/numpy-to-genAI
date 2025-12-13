"""
Exercise 3: Implement Your Own Convolution - Complete Solution

This script demonstrates a complete implementation of 2D convolution
applied to a real image (the project's panda mascot).

Author: Claude (NumPy-to-GenAI Project)
Date: 2025-01-07
"""

import numpy as np
from PIL import Image
import os

# =============================================================================
# Load the panda image
# =============================================================================
# Path to shared panda image (relative from this directory)
PANDA_PATH = '../../../3.3_artistic_filters/3.3.3_hexpanda/hexpanda/panda.png'

# Check if file exists, provide helpful error if not
if not os.path.exists(PANDA_PATH):
    print(f"Warning: Panda image not found at {PANDA_PATH}")
    print("Creating a synthetic test pattern instead...")

    # Create synthetic pattern as fallback
    SIZE = 256
    image = np.zeros((SIZE, SIZE), dtype=np.float64)
    # Add some circles and rectangles
    for y in range(SIZE):
        for x in range(SIZE):
            dist_center = np.sqrt((x - 128)**2 + (y - 128)**2)
            if dist_center < 60:
                image[y, x] = 255
            elif 80 < x < 180 and 20 < y < 80:
                image[y, x] = 200
else:
    # Load and convert to grayscale
    panda = Image.open(PANDA_PATH).convert('L')
    panda = panda.resize((256, 256))  # Resize for faster processing
    image = np.array(panda, dtype=np.float64)

print(f"Image loaded: {image.shape[0]}x{image.shape[1]} pixels")

# =============================================================================
# Define the edge detection kernel
# =============================================================================
edge_kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=np.float64)

print("\nEdge detection kernel:")
print(edge_kernel)

# =============================================================================
# The Complete Convolution Function
# =============================================================================
def apply_convolution(image, kernel):
    """
    Apply a convolution kernel to a grayscale image.

    The algorithm:
    1. Pad the image to handle border pixels
    2. For each pixel position (y, x):
       a. Extract the region under the kernel
       b. Multiply region by kernel (element-wise)
       c. Sum all values to get the output pixel

    Parameters:
        image (np.ndarray): 2D array of pixel values (grayscale)
        kernel (np.ndarray): 2D array of kernel weights (e.g., 3x3)

    Returns:
        np.ndarray: Convolved image (same size as input)
    """
    kernel_size = kernel.shape[0]
    pad = kernel_size // 2

    # Create output array (same size as input)
    height, width = image.shape
    output = np.zeros((height, width), dtype=np.float64)

    # Pad the image to handle borders
    # 'edge' mode repeats the edge pixels outward
    padded = np.pad(image, pad, mode='edge')

    # ==========================================================================
    # The core convolution loop - THIS IS THE KEY PART!
    # ==========================================================================
    for y in range(height):
        for x in range(width):
            # Extract the region that the kernel covers
            # Because we padded, we can safely access y:y+kernel_size
            region = padded[y:y + kernel_size, x:x + kernel_size]

            # Element-wise multiplication and sum
            # This IS convolution - multiply corresponding elements, then sum
            output[y, x] = np.sum(region * kernel)

    return output

# =============================================================================
# Apply the convolution
# =============================================================================
print("\nApplying convolution (this may take a moment)...")
result = apply_convolution(image, edge_kernel)
print("Convolution complete!")

# =============================================================================
# Post-process and save the result
# =============================================================================
# Edge detection can produce negative values and values > 255
# We need to handle this for display

# Option 1: Clip to valid range (simple but may lose information)
result_clipped = np.clip(result, 0, 255)

# Option 2: Normalize to 0-255 (preserves relative intensities)
result_min = result.min()
result_max = result.max()
if result_max > result_min:
    result_normalized = 255 * (result - result_min) / (result_max - result_min)
else:
    result_normalized = result_clipped

# Use normalized version for better visualization
final_result = result_normalized.astype(np.uint8)

# Save the result
output_image = Image.fromarray(final_result, mode='L')
output_image.save('convolution_solution.png')

print(f"\nResult saved as convolution_solution.png")
print(f"  Original pixel range: 0-255")
print(f"  After convolution: {result.min():.1f} to {result.max():.1f}")
print(f"  (Normalized back to 0-255 for display)")
print()
print("The edges of the panda should now be clearly visible as bright lines!")

# =============================================================================
# Bonus: Create a side-by-side comparison
# =============================================================================
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=150)

axes[0].imshow(image, cmap='gray', vmin=0, vmax=255)
axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(final_result, cmap='gray', vmin=0, vmax=255)
axes[1].set_title('Edge Detection Result', fontsize=12, fontweight='bold')
axes[1].axis('off')

plt.suptitle('Convolution in Action: Finding Edges',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('convolution_comparison.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("Comparison saved as convolution_comparison.png")
