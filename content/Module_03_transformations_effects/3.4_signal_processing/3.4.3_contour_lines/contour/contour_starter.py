"""
Exercise 3: Create Random Terrain Contours - Starter Code

Your task: Generate a random terrain with 3-5 hills at random positions
and create a contour visualization.

Instructions:
1. Complete the TODO sections below
2. Run the script to see your result
3. Try different random seeds to see different terrains

Author: Claude (NumPy-to-GenAI Project)
Date: 2025-12-07

Thesis Metadata:
    Framework: F1 (Hands-On Discovery)
    Module: 3.4 (Signal Processing)
    Exercise Type: Re-code (Create from scratch)
    Cognitive Load: Medium-High
"""

import numpy as np
from PIL import Image

# Set random seed for reproducibility (change this for different results)
np.random.seed(42)

# Step 1: Create coordinate grid (provided)
size = 400
dim = np.linspace(-10, 10, size)
x, y, _ = np.meshgrid(dim, dim, [1])

# Step 2: Generate random hills
# TODO: Set the number of hills (between 3 and 5)
num_hills = ___  # Replace ___ with a number

# TODO: Generate random positions for hill centers
# Hint: Use np.random.uniform(low, high, size) to get random values
# Positions should be within the grid bounds (-8 to 8 works well)
position_x = np.random.uniform(___, ___, num_hills)  # Fill in the blanks
position_y = np.random.uniform(___, ___, num_hills)  # Fill in the blanks

# TODO: Generate random widths for the hills
# Hint: Widths between 2.0 and 6.0 create nice visible hills
width_x = np.random.uniform(___, ___, num_hills)  # Fill in the blanks
width_y = np.random.uniform(___, ___, num_hills)  # Fill in the blanks

# Step 3: Calculate height using Gaussian formula (provided)
d = np.sqrt(((x - position_x) / width_x) ** 2 + ((y - position_y) / width_y) ** 2)
z = np.exp(-d ** 2)
z = z.sum(axis=2)

# Normalize to 0-1 range
znorm = (z - z.min()) / (z.max() - z.min())

# Step 4: Create contour visualization
# TODO: Choose number of contour levels (try values between 6 and 12)
n_levels = ___  # Replace ___ with a number

# Create stepped contour
step_size = 255 // n_levels
contour = (znorm * n_levels).astype(np.uint8) * step_size

# Save the result
image = Image.fromarray(contour, mode='L')
image.save('random_terrain.png')

print(f"Created random terrain with {num_hills} hills")
print(f"Using {n_levels} contour levels")
print("Saved as random_terrain.png")
