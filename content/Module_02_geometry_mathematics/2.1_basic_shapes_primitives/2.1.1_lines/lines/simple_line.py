"""
Exercise 2.1.1: Simple Line Drawing

Draws a single diagonal line to demonstrate the basic line drawing algorithm.
This exercise introduces learners to line interpolation using NumPy's linspace.

Author: Claude (NumPy-to-GenAI Project)
Date: 2025-01-30

Thesis Metadata:
    Framework: F1 (Hands-On Discovery)
    Module: Module_02_geometry_mathematics
    Exercise Type: Execute (Exercise 1)
    Cognitive Load: LOW
    New Concepts: Line interpolation, discrete pixel representation, parametric coordinates
    Prerequisites: Module 1.1.1 (NumPy arrays), Module 0.3.1 (array indexing)

Research Question Contributions:
    RQ1 (Framework Design): Demonstrates immediate visual feedback pattern
    RQ2 (Cognitive Load): Builds on prior array knowledge, introduces 1 new concept
    RQ4 (Assessment): Technical dimension - execute code successfully
    RQ5 (Transfer): Parametric thinking transfers to curves, transformations

Learning Objectives:
    - Understand how continuous lines are represented discretely
    - Use NumPy's linspace for coordinate interpolation
    - Recognize the need for integer coordinates in pixel arrays
"""

import numpy as np
from PIL import Image

# Step 1: Create blank canvas (grayscale image)
canvas = np.zeros((400, 400), dtype=np.uint8)

# Step 2: Define line endpoints
x_start, y_start = 50, 50
x_end, y_end = 350, 350

# Step 3: Calculate number of points needed
# We need at least one point per pixel in the longer dimension
num_points = max(abs(x_end - x_start), abs(y_end - y_start)) + 1

# Step 4: Generate interpolated coordinates using linspace
# linspace creates evenly spaced points between start and end
x_coords = np.linspace(x_start, x_end, num_points).round().astype(int)
y_coords = np.linspace(y_start, y_end, num_points).round().astype(int)

# Step 5: Draw line by setting pixels to white (255)
# Remember: array indexing is [row, column] which is [y, x]
canvas[y_coords, x_coords] = 255

# Step 6: Save result
output_image = Image.fromarray(canvas)
output_image.save('simple_line.png')
print("Simple line saved as simple_line.png")
print(f"Line drawn from ({x_start}, {y_start}) to ({x_end}, {y_end})")
print(f"Using {num_points} interpolated points")
