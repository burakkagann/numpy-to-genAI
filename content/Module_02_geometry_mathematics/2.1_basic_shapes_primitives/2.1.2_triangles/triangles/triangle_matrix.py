"""
Exercise 2.1.2: Triangle using Matrix Operations

This script demonstrates an elegant mathematical approach to creating triangles
using vector transpose operations. When we add a column vector to its transpose
(a row vector), we create a 2D matrix where each value represents a sum of
coordinates - this naturally creates diagonal boundaries.

Author: Claude (NumPy-to-GenAI Project)
Date: 2025-01-30

Thesis Metadata:
    Framework: F1-Hands-On
    Module: Module_02_geometry_mathematics
    Exercise Type: Core Concept 2 demonstration
    Cognitive Load: Medium
    New Concepts: vector transpose, matrix broadcasting, conditional masking

Key Insight:
    The expression `x + x.T` where x is a column vector creates a matrix where
    element [i,j] = i + j. This creates natural diagonal lines (i + j = constant).
"""

import numpy as np
from PIL import Image

# Step 1: Create a column vector of values 0 to 399
# reshape(400, 1) turns the 1D array into a 2D column vector
x = np.arange(400).reshape(400, 1)

# Step 2: Create the distance matrix using transpose
# x has shape (400, 1), x.T has shape (1, 400)
# NumPy broadcasts to create a (400, 400) matrix where element [i,j] = i + j
distance_matrix = x + x.T

# Step 3: Apply conditional masking to create triangle
# Points where i + j <= 400 form a triangle in the upper-left corner
# This creates a diagonal boundary line from (0, 400) to (400, 0)
threshold = 400
triangle_mask = distance_matrix <= threshold

# Step 4: Create the image
# Convert boolean mask to uint8 (True->255, False->0)
canvas = np.zeros((400, 400), dtype=np.uint8)
canvas[triangle_mask] = 255

# Save result
Image.fromarray(canvas).save('triangle_matrix.png')

print("Matrix-based triangle created successfully!")
print(f"Output saved as: triangle_matrix.png")
print(f"Canvas dimensions: {canvas.shape}")
print(f"Threshold used: {threshold}")
print(f"Triangle boundary: diagonal line where row + col = {threshold}")
print()
print("Key insight: x + x.T creates a matrix where each element equals")
print("the sum of its row and column indices. This naturally creates")
print("diagonal boundaries for geometric shapes!")
