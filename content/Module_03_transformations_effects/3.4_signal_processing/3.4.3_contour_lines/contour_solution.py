"""
Exercise 3: Create Random Terrain Contours - Complete Solution

Generates a random terrain with multiple hills at random positions
and creates both stepped contour and isoline visualizations.

Author: Claude (NumPy-to-GenAI Project)
Date: 2025-12-07

Thesis Metadata:
    Framework: F1 (Hands-On Discovery)
    Module: 3.4 (Signal Processing)
    Exercise Type: Re-code Solution
    Cognitive Load: Medium-High
    Core Concepts: random generation, Gaussian mixture, contours
    Prerequisites: contour.py execution and modification exercises

RQ Contributions:
    RQ1: Demonstrates mastery through creative application
    RQ2: Builds on prior concepts without overload
    RQ4: Clear success criteria (visual output matches expectations)
    RQ5: Transfers learned concepts to novel terrain generation

Learning Objectives:
    - Apply np.random to generate varied terrain parameters
    - Combine multiple Gaussians with random parameters
    - Create contour visualizations from generated data
"""

import numpy as np
from PIL import Image

# Set random seed for reproducibility (change for different results)
np.random.seed(42)

# Step 1: Create coordinate grid
size = 400
dim = np.linspace(-10, 10, size)
x, y, _ = np.meshgrid(dim, dim, [1])

# Step 2: Generate random hills (between 3 and 5)
num_hills = 4  # You could also use: np.random.randint(3, 6)

# Random positions within the grid (avoid edges)
position_x = np.random.uniform(-8, 8, num_hills)
position_y = np.random.uniform(-8, 8, num_hills)

# Random widths (2.0 to 6.0 creates nice visible hills)
width_x = np.random.uniform(2.0, 6.0, num_hills)
width_y = np.random.uniform(2.0, 6.0, num_hills)

print(f"Generated {num_hills} hills:")
for i in range(num_hills):
    print(f"  Hill {i+1}: position=({position_x[i]:.1f}, {position_y[i]:.1f}), "
          f"width=({width_x[i]:.1f}, {width_y[i]:.1f})")

# Step 3: Calculate height using Gaussian formula
# Each hill contributes to the total height
d = np.sqrt(((x - position_x) / width_x) ** 2 + ((y - position_y) / width_y) ** 2)
z = np.exp(-d ** 2)
z = z.sum(axis=2)  # Combine all hills

# Normalize to 0-1 range
znorm = (z - z.min()) / (z.max() - z.min())

# Step 4: Create stepped contour visualization
n_levels = 10  # More levels for finer detail
step_size = 255 // n_levels
contour = (znorm * n_levels).astype(np.uint8) * step_size

# Save stepped contour
image = Image.fromarray(contour, mode='L')
image.save('random_terrain.png')
print(f"\nSaved random_terrain.png with {n_levels} contour levels")

# Bonus: Also create isoline version
isolines = ((znorm * 100).round() % 12) == 0
isolines = (isolines * 255).astype(np.uint8)
image = Image.fromarray(isolines, mode='L')
image.save('random_terrain_isolines.png')
print("Saved random_terrain_isolines.png (isoline version)")

print("\nDone! Your random terrain has been generated.")
