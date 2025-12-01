"""
Exercise 2.2.1: Simple Horizontal Gradient

Creates a horizontal grayscale gradient using np.linspace() to generate
evenly spaced values across the image width.

Author: Claude (NumPy-to-GenAI Project)
Date: 2025-01-30

Thesis Metadata:
    Framework: F1-Hands-On
    Module: Module_02_geometry_mathematics
    Exercise Type: Quick Start / Execute
    Cognitive Load: Low
    New Concepts: np.linspace(), gradient creation, broadcasting
    Prerequisites: Module 1.1.1 (RGB basics), Module 2.1 (basic shapes)

Research Question Contributions:
    RQ1: Visual-first approach - immediate gradient output before theory
    RQ2: Single concept introduction (linear interpolation)
"""

import numpy as np
from PIL import Image

# Step 1: Define image dimensions (rectangular to show gradient direction)
height = 300
width = 800

# Step 2: Create a 1D array of evenly spaced values from 0 (black) to 255 (white)
# np.linspace(start, stop, num) generates 'num' values between start and stop
gradient_values = np.linspace(0, 255, width, dtype=np.uint8)

# Step 3: Create the full 2D grayscale image by repeating the gradient for each row
# Broadcasting: a 1D array of shape (width,) becomes (height, width)
gradient_image = np.tile(gradient_values, (height, 1))

# Step 4: Save the gradient image
output = Image.fromarray(gradient_image, mode='L')  # 'L' mode for grayscale
output.save('simple_gradient.png')

print(f"Created horizontal gradient: {width}x{height} pixels")
print(f"Value range: {gradient_values[0]} (left) to {gradient_values[-1]} (right)")
print("Image saved as simple_gradient.png")
