"""
Convolution Concept Diagram

This script creates an educational diagram showing how 2D convolution works:
- A kernel overlays a region of the image
- Element-wise multiplication occurs
- Results are summed to produce one output pixel

Author: Claude (NumPy-to-GenAI Project)
Date: 2025-01-07
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# =============================================================================
# Create the diagram
# =============================================================================
fig, axes = plt.subplots(1, 4, figsize=(14, 4), dpi=150)

# Sample image region (5x5 for visibility)
image_region = np.array([
    [10, 20, 30, 40, 50],
    [15, 25, 35, 45, 55],
    [20, 30, 40, 50, 60],
    [25, 35, 45, 55, 65],
    [30, 40, 50, 60, 70]
], dtype=float)

# 3x3 blur kernel (we'll highlight the center 3x3)
kernel = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
], dtype=float) / 9.0

# =============================================================================
# Panel 1: Image Region
# =============================================================================
ax1 = axes[0]
ax1.imshow(image_region, cmap='gray', vmin=0, vmax=100)
ax1.set_title('Image Region\n(5x5 pixels)', fontsize=11, fontweight='bold')

# Add pixel values as text
for i in range(5):
    for j in range(5):
        ax1.text(j, i, f'{int(image_region[i, j])}',
                ha='center', va='center', fontsize=8, color='yellow')

# Highlight the 3x3 area where kernel will be applied
rect = patches.Rectangle((0.5, 0.5), 3, 3, linewidth=3,
                         edgecolor='red', facecolor='none')
ax1.add_patch(rect)
ax1.set_xticks([])
ax1.set_yticks([])

# =============================================================================
# Panel 2: Kernel
# =============================================================================
ax2 = axes[1]
ax2.imshow(kernel, cmap='Blues', vmin=0, vmax=0.2)
ax2.set_title('Kernel\n(3x3 weights)', fontsize=11, fontweight='bold')

# Add kernel values
for i in range(3):
    for j in range(3):
        ax2.text(j, i, f'{kernel[i, j]:.2f}',
                ha='center', va='center', fontsize=9, color='black',
                fontweight='bold')

ax2.set_xticks([])
ax2.set_yticks([])

# Add multiplication symbol between panels
fig.text(0.32, 0.5, '×', fontsize=30, ha='center', va='center', fontweight='bold')

# =============================================================================
# Panel 3: Element-wise multiplication
# =============================================================================
ax3 = axes[2]

# Extract the center 3x3 region that the kernel covers
center_region = image_region[1:4, 1:4]
multiplied = center_region * kernel

ax3.imshow(multiplied, cmap='Greens', vmin=0, vmax=10)
ax3.set_title('Multiply\n(pixel × weight)', fontsize=11, fontweight='bold')

# Add multiplied values
for i in range(3):
    for j in range(3):
        ax3.text(j, i, f'{multiplied[i, j]:.1f}',
                ha='center', va='center', fontsize=9, color='black',
                fontweight='bold')

ax3.set_xticks([])
ax3.set_yticks([])

# Add equals symbol
fig.text(0.72, 0.5, '=', fontsize=30, ha='center', va='center', fontweight='bold')

# =============================================================================
# Panel 4: Sum (output pixel)
# =============================================================================
ax4 = axes[3]

output_value = multiplied.sum()

# Create a single cell visualization
single_pixel = np.array([[output_value]])
ax4.imshow(single_pixel, cmap='Oranges', vmin=0, vmax=50)
ax4.set_title('Sum\n(output pixel)', fontsize=11, fontweight='bold')
ax4.text(0, 0, f'{output_value:.1f}',
        ha='center', va='center', fontsize=14, color='black',
        fontweight='bold')
ax4.set_xticks([])
ax4.set_yticks([])

# =============================================================================
# Add overall title and annotations
# =============================================================================
fig.suptitle('How 2D Convolution Works', fontsize=14, fontweight='bold', y=1.02)

# Add step labels below
fig.text(0.14, 0.02, 'Step 1: Select region', ha='center', fontsize=9)
fig.text(0.38, 0.02, 'Step 2: Apply kernel', ha='center', fontsize=9)
fig.text(0.62, 0.02, 'Step 3: Multiply', ha='center', fontsize=9)
fig.text(0.86, 0.02, 'Step 4: Sum', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('convolution_concept.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("Convolution concept diagram saved as convolution_concept.png")
print()
print("The diagram shows:")
print("  1. A 5x5 image region with the 3x3 target area highlighted")
print("  2. A 3x3 averaging kernel (all values = 1/9)")
print("  3. Element-wise multiplication results")
print(f"  4. Final sum = {output_value:.1f} (the output pixel value)")
