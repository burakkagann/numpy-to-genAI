import numpy as np
from PIL import Image


def draw_fractal_square(canvas, x_min, x_max, y_min, y_max, depth):
    """
    Recursively draw fractal squares on the canvas.

    Parameters
    ----------
    canvas : numpy.ndarray
        The image array to draw on (height, width, 3)
    x_min, x_max : int
        Horizontal bounds of the current region
    y_min, y_max : int
        Vertical bounds of the current region
    depth : int
        Remaining recursion depth (stops when 0)
    """
    # Calculate the boundaries for the 3x3 grid
    # The region is divided into thirds both horizontally and vertically
    x_third = (x_max - x_min) // 3
    y_third = (y_max - y_min) // 3

    # Define the center square boundaries
    center_x_start = x_min + x_third
    center_x_end = x_min + 2 * x_third
    center_y_start = y_min + y_third
    center_y_end = y_min + 2 * y_third

    # Fill the center square with green color
    # Using += creates an accumulation effect for overlapping regions
    canvas[center_y_start:center_y_end, center_x_start:center_x_end, 1] += 32

    # If we haven't reached the base case, recurse into the four corners
    if depth > 0:
        # Top-left corner
        draw_fractal_square(canvas, x_min, center_x_end, y_min, center_y_end, depth - 1)

        # Top-right corner
        draw_fractal_square(canvas, center_x_start, x_max, y_min, center_y_end, depth - 1)

        # Bottom-left corner
        draw_fractal_square(canvas, x_min, center_x_end, center_y_start, y_max, depth - 1)

        # Bottom-right corner
        draw_fractal_square(canvas, center_x_start, x_max, center_y_start, y_max, depth - 1)


# Create the canvas
canvas_size = 800
canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

# Generate the fractal with 3 levels of recursion
recursion_depth = 3
draw_fractal_square(canvas, 0, canvas_size, 0, canvas_size, recursion_depth)

# Save the result
image = Image.fromarray(canvas)
image.save("fractal_square.png")
print(f"Saved fractal_square.png ({canvas_size}x{canvas_size}, depth={recursion_depth})")
