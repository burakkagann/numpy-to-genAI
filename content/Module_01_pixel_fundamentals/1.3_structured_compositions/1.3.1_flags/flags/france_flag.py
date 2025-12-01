"""
Exercise 1.3.1: Flags (Array Slicing) - France Flag

Teaches NumPy array slicing to create vertical color bands.
This is a simple demonstration of column slicing using the [:, start:stop] notation.

Author: Claude (NumPy-to-GenAI Project)
Date: 2025-01-30

Thesis Metadata:
    Framework: F1-Hands-On
    Module: Module_01_pixel_fundamentals
    Exercise Type: Execute
    Cognitive Load: Low
    New Concepts: [array slicing notation, column slicing, rectangular regions]
    Prerequisites: [RGB basics (1.1.1), basic NumPy arrays]

Research Question Contributions:
    RQ1 (Framework Design): Demonstrates visual-first array slicing with immediate feedback
    RQ2 (Cognitive Load): Simple concept (vertical slicing) with clear visual validation
    RQ4 (Assessment): Technical (correct slicing syntax), Creative (accurate flag colors), Conceptual (understanding [:, start:stop])
    RQ5 (Transfer): Column slicing transfers to tiling patterns (Module 1.3.2-1.3.4)

Learning Objectives:
    - Understand column slicing notation [:, start:stop]
    - Use colon operator to select all rows
    - Create structured compositions with rectangular regions
"""

import numpy as np
from PIL import Image

# Step 1: Create blank canvas (standard 2:3 flag ratio)
height, width = 300, 450
flag = np.zeros((height, width, 3), dtype=np.uint8)

# Step 2: Create three vertical bands using column slicing
# Blue stripe (left third: columns 0-149)
flag[:, 0:150, 0] = 0    # Red channel
flag[:, 0:150, 1] = 85   # Green channel
flag[:, 0:150, 2] = 164  # Blue channel (French blue #0055A4)

# White stripe (middle third: columns 150-299)
flag[:, 150:300, :] = 255  # All channels set to 255 for white

# Red stripe (right third: columns 300-449)
flag[:, 300:450, 0] = 239  # Red channel (French red #EF4135)
flag[:, 300:450, 1] = 65   # Green channel
flag[:, 300:450, 2] = 53   # Blue channel

# Step 3: Save the flag
result = Image.fromarray(flag, mode='RGB')
result.save('france_flag.png')

print("France flag created successfully!")
print(f"Output saved as: france_flag.png")
print(f"Flag dimensions: {flag.shape} (height, width, channels)")
