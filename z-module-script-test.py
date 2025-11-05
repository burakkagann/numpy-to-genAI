import numpy as np
from PIL import Image
from scipy.ndimage import convolve

def grid_to_image(grid, scale=10):
    """Convert binary grid to RGB image."""
    gray = np.repeat(np.repeat(grid * 255, scale, axis=0), scale, axis=1)
    return np.stack([gray, gray, gray], axis=2).astype(np.uint8)

def game_of_life_step(grid):
    """Apply one generation of Conway's Game of Life."""
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    neighbor_count = convolve(grid, kernel, mode='wrap')
    return ((neighbor_count == 3) | ((grid == 1) & (neighbor_count == 2))).astype(int)

# Initialize with blinker pattern (horizontal line of 3 cells)
grid = np.zeros((30, 30), dtype=int)
grid[15, 14:17] = [1, 1, 1]  # Horizontal blinker in center

# Evolve for 6 generations to see complete cycles
for generation in range(6):
    grid = game_of_life_step(grid)
    print(f"Generation {generation + 1}: {np.sum(grid)} living cells")

# Save final state
Image.fromarray(grid_to_image(grid)).save('exercise1_result.png')
print("Blinker oscillator evolution completed!")