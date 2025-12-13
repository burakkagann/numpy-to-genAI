"""
Exercise 3.4.2: Edge Detection Operator Comparison

This script compares three classic edge detection operators:
Roberts Cross, Prewitt, and Sobel on the same test image.

Author: Claude (NumPy-to-GenAI Project)
Date: 2025-12-07

Thesis Metadata:
    Framework: F1-Hands-On
    Module: Module_03_transformations_effects
    Exercise Type: Conceptual Comparison
    Cognitive Load: Medium
    New Concepts: Roberts, Prewitt, Operator comparison
    Prerequisites: Sobel operator, Convolution
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Step 1: Create a test image with varied features
height, width = 200, 200
image = np.zeros((height, width), dtype=np.float64)

# Draw a rectangle
image[40:80, 30:90] = 255

# Draw a circle
center_y, center_x, radius = 120, 60, 35
y_coords, x_coords = np.ogrid[:height, :width]
circle_mask = (x_coords - center_x)**2 + (y_coords - center_y)**2 <= radius**2
image[circle_mask] = 200

# Draw diagonal lines
for i in range(50):
    if 130 + i < height and 110 + i < width:
        image[130 + i, 110 + i] = 255
        image[131 + i, 110 + i] = 255
    if 80 + i < height and 170 - i >= 0 and 170 - i < width:
        image[80 + i, 170 - i] = 255
        image[81 + i, 170 - i] = 255

# Add a gradient region
image[20:60, 130:190] = np.tile(np.linspace(0, 255, 60), (40, 1))

# Step 2: Define all edge detection kernels

# Roberts Cross (2x2 kernels)
roberts_gx = np.array([[1, 0],
                       [0, -1]])
roberts_gy = np.array([[0, 1],
                       [-1, 0]])

# Prewitt (3x3 kernels, unweighted)
prewitt_gx = np.array([[-1, 0, 1],
                       [-1, 0, 1],
                       [-1, 0, 1]])
prewitt_gy = np.array([[-1, -1, -1],
                       [ 0,  0,  0],
                       [ 1,  1,  1]])

# Sobel (3x3 kernels, weighted center)
sobel_gx = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])
sobel_gy = np.array([[-1, -2, -1],
                     [ 0,  0,  0],
                     [ 1,  2,  1]])


def apply_roberts(img):
    """Apply Roberts Cross edge detection."""
    h, w = img.shape
    result = np.zeros((h, w), dtype=np.float64)
    for y in range(h - 1):
        for x in range(w - 1):
            neighborhood = img[y:y+2, x:x+2]
            gx = np.sum(roberts_gx * neighborhood)
            gy = np.sum(roberts_gy * neighborhood)
            result[y, x] = np.sqrt(gx**2 + gy**2)
    return result


def apply_3x3_operator(img, kernel_gx, kernel_gy):
    """Apply a 3x3 edge detection operator (Prewitt or Sobel)."""
    h, w = img.shape
    result = np.zeros((h, w), dtype=np.float64)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            neighborhood = img[y-1:y+2, x-1:x+2]
            gx = np.sum(kernel_gx * neighborhood)
            gy = np.sum(kernel_gy * neighborhood)
            result[y, x] = np.sqrt(gx**2 + gy**2)
    return result


# Step 3: Apply all operators
roberts_result = apply_roberts(image)
prewitt_result = apply_3x3_operator(image, prewitt_gx, prewitt_gy)
sobel_result = apply_3x3_operator(image, sobel_gx, sobel_gy)

# Normalize all results to 0-255
def normalize(arr):
    if arr.max() > 0:
        return (255 * arr / arr.max()).astype(np.uint8)
    return arr.astype(np.uint8)

roberts_normalized = normalize(roberts_result)
prewitt_normalized = normalize(prewitt_result)
sobel_normalized = normalize(sobel_result)

# Step 4: Create comparison figure
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Original image
axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

# Roberts result
axes[0, 1].imshow(roberts_normalized, cmap='gray')
axes[0, 1].set_title('Roberts Cross\n(2x2 kernels, fast but noisy)', fontsize=11, fontweight='bold')
axes[0, 1].axis('off')

# Prewitt result
axes[1, 0].imshow(prewitt_normalized, cmap='gray')
axes[1, 0].set_title('Prewitt\n(3x3 kernels, unweighted)', fontsize=11, fontweight='bold')
axes[1, 0].axis('off')

# Sobel result
axes[1, 1].imshow(sobel_normalized, cmap='gray')
axes[1, 1].set_title('Sobel\n(3x3 kernels, weighted center)', fontsize=11, fontweight='bold')
axes[1, 1].axis('off')

plt.suptitle('Edge Detection Operator Comparison', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('operator_comparison.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("Operator comparison created!")
print("Output saved as: operator_comparison.png")
print(f"\nMaximum edge strengths:")
print(f"  Roberts: {roberts_result.max():.2f}")
print(f"  Prewitt: {prewitt_result.max():.2f}")
print(f"  Sobel:   {sobel_result.max():.2f}")
