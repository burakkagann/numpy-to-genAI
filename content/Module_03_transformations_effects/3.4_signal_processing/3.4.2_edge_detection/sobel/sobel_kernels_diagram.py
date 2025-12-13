"""
Exercise 3.4.2: Sobel Kernels Visualization

This script generates a conceptual diagram showing the Sobel Gx and Gy
kernels with color-coded values and directional annotations.

Author: Claude (NumPy-to-GenAI Project)
Date: 2025-12-07

Thesis Metadata:
    Framework: F1-Hands-On
    Module: Module_03_transformations_effects
    Exercise Type: Conceptual Diagram
    Cognitive Load: Low
    New Concepts: Kernel visualization, Gradient direction
    Prerequisites: NumPy arrays
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the Sobel kernels
sobel_gx = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])

sobel_gy = np.array([[-1, -2, -1],
                     [ 0,  0,  0],
                     [ 1,  2,  1]])

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Color map: negative=blue, zero=white, positive=red
cmap = 'RdBu_r'
vmin, vmax = -2, 2

# Plot Gx kernel (detects vertical edges)
ax1 = axes[0]
im1 = ax1.imshow(sobel_gx, cmap=cmap, vmin=vmin, vmax=vmax)
ax1.set_title('Sobel Gx Kernel\n(Detects Vertical Edges)', fontsize=14, fontweight='bold')

# Add numerical values to cells
for i in range(3):
    for j in range(3):
        color = 'white' if abs(sobel_gx[i, j]) > 1 else 'black'
        ax1.text(j, i, str(sobel_gx[i, j]), ha='center', va='center',
                fontsize=18, fontweight='bold', color=color)

ax1.set_xticks([0, 1, 2])
ax1.set_yticks([0, 1, 2])
ax1.set_xticklabels(['Left', 'Center', 'Right'])
ax1.set_yticklabels(['Top', 'Middle', 'Bottom'])

# Add arrow showing detection direction
ax1.annotate('', xy=(2.6, 1), xytext=(-0.6, 1),
            arrowprops=dict(arrowstyle='->', color='green', lw=3))
ax1.text(1, 2.8, 'Measures horizontal change', ha='center', fontsize=11,
         style='italic', color='darkgreen')

# Plot Gy kernel (detects horizontal edges)
ax2 = axes[1]
im2 = ax2.imshow(sobel_gy, cmap=cmap, vmin=vmin, vmax=vmax)
ax2.set_title('Sobel Gy Kernel\n(Detects Horizontal Edges)', fontsize=14, fontweight='bold')

# Add numerical values to cells
for i in range(3):
    for j in range(3):
        color = 'white' if abs(sobel_gy[i, j]) > 1 else 'black'
        ax2.text(j, i, str(sobel_gy[i, j]), ha='center', va='center',
                fontsize=18, fontweight='bold', color=color)

ax2.set_xticks([0, 1, 2])
ax2.set_yticks([0, 1, 2])
ax2.set_xticklabels(['Left', 'Center', 'Right'])
ax2.set_yticklabels(['Top', 'Middle', 'Bottom'])

# Add arrow showing detection direction
ax2.annotate('', xy=(1, 2.6), xytext=(1, -0.6),
            arrowprops=dict(arrowstyle='->', color='green', lw=3))
ax2.text(1, 2.8, 'Measures vertical change', ha='center', fontsize=11,
         style='italic', color='darkgreen')

# Add colorbar
cbar = fig.colorbar(im1, ax=axes, orientation='horizontal',
                    fraction=0.05, pad=0.15, shrink=0.6)
cbar.set_label('Kernel Weight (negative=subtract, positive=add)', fontsize=11)

plt.tight_layout()
plt.savefig('sobel_kernels_diagram.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("Sobel kernels diagram created!")
print("Output saved as: sobel_kernels_diagram.png")
