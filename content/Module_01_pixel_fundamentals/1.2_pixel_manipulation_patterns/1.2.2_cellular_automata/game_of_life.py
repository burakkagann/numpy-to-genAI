"""
Exercise 1.2.2: Cellular Automata - Conway's Game of Life

Discover how simple rules create complex, evolving patterns.
This demonstrates emergent behavior in computational systems.

Author: Claude (NumPy-to-GenAI Project)
Date: 2024-10-31
"""

import numpy as np
from PIL import Image
from scipy.ndimage import convolve

# Initialize 50x50 grid with a "glider" pattern
grid = np.zeros((50, 50), dtype=int)
grid[20:23, 20:23] = [[0, 1, 0], [0, 0, 1], [1, 1, 1]]  # Classic glider

# Define Game of Life rules using convolution
kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])  # Count 8 neighbors
neighbor_count = convolve(grid, kernel, mode='wrap')

# Apply Conway's rules: Birth (3 neighbors), Survival (2-3 neighbors)
grid = ((neighbor_count == 3) | ((grid == 1) & (neighbor_count == 2))).astype(int)

# Convert to image (scale up and colorize)
image_array = np.repeat(np.repeat(grid * 255, 8, axis=0), 8, axis=1)
image_array = np.stack([image_array, image_array, image_array], axis=2).astype(np.uint8)

# Save result
result_image = Image.fromarray(image_array)
result_image.save('game_of_life_step.png')

print("Game of Life step completed!")
print(f"Grid size: {grid.shape}")
print(f"Living cells: {np.sum(grid)}")
print(f"Image size: {image_array.shape}")