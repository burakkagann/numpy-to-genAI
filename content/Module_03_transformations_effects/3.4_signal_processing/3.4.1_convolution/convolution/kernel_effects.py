"""
Kernel Effects Comparison

This script demonstrates how different kernels produce different effects:
- Identity: No change (passes through original)
- Blur: Smooths edges by averaging neighbors
- Sharpen: Enhances edges by emphasizing center pixel
- Edge Detection: Highlights boundaries between light and dark

Author: Claude (NumPy-to-GenAI Project)
Date: 2025-01-07
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# =============================================================================
# Step 1: Create a synthetic test image with various features
# =============================================================================
SIZE = 200
canvas = np.zeros((SIZE, SIZE), dtype=np.float64)

# Add a white rectangle
canvas[40:80, 30:90] = 255

# Add a white circle
center_y, center_x = 120, 60
for y in range(SIZE):
    for x in range(SIZE):
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        if distance < 30:
            canvas[y, x] = 255

# Add a gradient region
canvas[40:160, 120:180] = np.tile(np.linspace(0, 255, 60), (120, 1))

# Add some diagonal lines
for i in range(0, SIZE, 20):
    for offset in range(min(SIZE - i, SIZE)):
        if i + offset < SIZE and offset < SIZE:
            canvas[i + offset, offset] = 200

# =============================================================================
# Step 2: Define different kernels
# =============================================================================
kernels = {
    'Original': None,  # Special case - no convolution

    'Blur (Box)': np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ], dtype=np.float64) / 9.0,

    'Sharpen': np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float64),

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
    """Apply a kernel to an image using convolution."""
    if kernel is None:
        return image.copy()

    kernel_size = kernel.shape[0]
    pad = kernel_size // 2

    # Output will be same size as input (we'll handle borders)
    output = np.zeros_like(image)

    # Pad the image to handle borders
    padded = np.pad(image, pad, mode='edge')

    height, width = image.shape
    for y in range(height):
        for x in range(width):
            # Extract region and apply kernel
            region = padded[y:y + kernel_size, x:x + kernel_size]
            output[y, x] = np.sum(region * kernel)

    return output

# =============================================================================
# Step 4: Apply each kernel and create comparison grid
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=150)
axes = axes.flatten()

for idx, (name, kernel) in enumerate(kernels.items()):
    # Apply convolution
    result = apply_convolution(canvas, kernel)

    # Clip values to valid range
    result = np.clip(result, 0, 255)

    # Display
    axes[idx].imshow(result, cmap='gray', vmin=0, vmax=255)
    axes[idx].set_title(name, fontsize=14, fontweight='bold', pad=10)
    axes[idx].axis('off')

    # Add kernel visualization for non-original
    if kernel is not None:
        # Create a small text showing kernel values
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
