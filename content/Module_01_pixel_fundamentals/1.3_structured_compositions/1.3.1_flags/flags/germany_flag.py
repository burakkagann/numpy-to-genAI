"""
Exercise 1.3.1: Flags (Array Slicing) - Germany Flag

Demonstrates horizontal slicing to create row-based color bands.
This is the solution for Exercise 2 (Modify phase).

Author: Claude (NumPy-to-GenAI Project)
Date: 2025-01-30

Thesis Metadata:
    Framework: F1-Hands-On
    Module: Module_01_pixel_fundamentals
    Exercise Type: Modify
    Cognitive Load: Low
    New Concepts: [row slicing, horizontal bands]
    Prerequisites: [Exercise 1 (France flag), column slicing]

Research Question Contributions:
    RQ1 (Framework Design): Shows transfer from column to row slicing
    RQ2 (Cognitive Load): Builds on previous knowledge (column slicing → row slicing)
    RQ4 (Assessment): Tests ability to transfer slicing pattern to different dimension
    RQ5 (Transfer): Reinforces dimensional understanding for future transformations

Learning Objectives:
    - Apply row slicing notation [start:stop, :]
    - Understand dimension differences (rows vs columns)
    - Transfer vertical slicing knowledge to horizontal slicing
"""

import numpy as np
from PIL import Image

# Step 1: Create blank canvas (standard 2:3 flag ratio)
height, width = 300, 450
flag = np.zeros((height, width, 3), dtype=np.uint8)

# Step 2: Create three horizontal bands using row slicing
# Black stripe (top third: rows 0-99)
flag[0:100, :, :] = [0, 0, 0]  # Black

# Red stripe (middle third: rows 100-199)
flag[100:200, :, :] = [221, 0, 0]  # German red (#DD0000)

# Gold stripe (bottom third: rows 200-299)
flag[200:300, :, :] = [255, 206, 0]  # German gold (#FFCE00)

# Step 3: Save the flag
result = Image.fromarray(flag, mode='RGB')
result.save('germany_flag.png')

print("Germany flag created successfully!")
print(f"Output saved as: germany_flag.png")
print(f"Flag dimensions: {flag.shape} (height, width, channels)")
