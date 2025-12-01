"""
Exercise 1.3.1: Flags (Array Slicing) - Switzerland Flag (Complete Solution)

Demonstrates combining horizontal and vertical slicing to create complex patterns.
This is the complete solution for Exercise 3 (Re-code phase).

Author: Claude (NumPy-to-GenAI Project)
Date: 2025-01-30

Thesis Metadata:
    Framework: F1-Hands-On
    Module: Module_01_pixel_fundamentals
    Exercise Type: Re-code
    Cognitive Load: Medium
    New Concepts: [combining slices, centering elements, overlapping rectangles]
    Prerequisites: [column slicing (Exercise 1), row slicing (Exercise 2)]

Research Question Contributions:
    RQ1 (Framework Design): Demonstrates synthesis of vertical and horizontal slicing
    RQ2 (Cognitive Load): Requires combining two previous concepts (manageable complexity)
    RQ4 (Assessment): Tests conceptual understanding (cross = overlapping rectangles) and technical skill (centering calculations)
    RQ5 (Transfer): Prepares for complex compositions in tiling (Module 1.3.2-1.3.4)

Learning Objectives:
    - Combine horizontal and vertical slicing in single composition
    - Calculate centered positions using integer division
    - Understand overlapping rectangular regions
"""

import numpy as np
from PIL import Image

# Step 1: Create square flag (Switzerland uses 1:1 ratio)
size = 300
flag = np.zeros((size, size, 3), dtype=np.uint8)

# Step 2: Fill entire background with red
flag[:, :] = [255, 0, 0]  # Swiss red

# Step 3: Create white vertical bar (centered)
# Bar width is 1/5 of total size
bar_width = size // 5  # 60 pixels
left = (size - bar_width) // 2  # 120
right = left + bar_width  # 180
flag[:, left:right, :] = 255  # White vertical bar

# Step 4: Create white horizontal bar (centered)
# Bar height is 1/5 of total size
bar_height = size // 5  # 60 pixels
top = (size - bar_height) // 2  # 120
bottom = top + bar_height  # 180
flag[top:bottom, :, :] = 255  # White horizontal bar

# Step 5: Save the flag
result = Image.fromarray(flag, mode='RGB')
result.save('switzerland_flag.png')

print("Switzerland flag created successfully!")
print(f"Output saved as: switzerland_flag.png")
print(f"Flag dimensions: {flag.shape} (height, width, channels)")
print(f"Cross dimensions: {bar_height}x{bar_width} pixels")
