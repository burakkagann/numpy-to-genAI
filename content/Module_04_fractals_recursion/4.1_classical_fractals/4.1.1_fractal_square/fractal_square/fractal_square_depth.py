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
    # Calculate thirds for the 3x3 grid division
    x_third = (x_max - x_min) // 3
    y_third = (y_max - y_min) // 3

    # Center square boundaries
    center_x_start = x_min + x_third
    center_x_end = x_min + 2 * x_third
    center_y_start = y_min + y_third
    center_y_end = y_min + 2 * y_third

    # Fill center with green (accumulating effect)
    canvas[center_y_start:center_y_end, center_x_start:center_x_end, 1] += 32

    # Recurse into four corners if depth allows
    if depth > 0:
        draw_fractal_square(canvas, x_min, center_x_end, y_min, center_y_end, depth - 1)
        draw_fractal_square(canvas, center_x_start, x_max, y_min, center_y_end, depth - 1)
        draw_fractal_square(canvas, x_min, center_x_end, center_y_start, y_max, depth - 1)
        draw_fractal_square(canvas, center_x_start, x_max, center_y_start, y_max, depth - 1)


def generate_fractal_at_depth(depth, canvas_size=800):
    """
    Generate a fractal square image at the specified depth.

    Parameters
    ----------
    depth : int
        The recursion depth (higher = more detail)
    canvas_size : int
        Width and height of the output image

    Returns
    -------
    numpy.ndarray
        The generated image array
    """
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    draw_fractal_square(canvas, 0, canvas_size, 0, canvas_size, depth)
    return canvas


# Generate images at different depths
depths_to_generate = [1, 2, 4]

for depth in depths_to_generate:
    # Create the fractal
    image_array = generate_fractal_at_depth(depth)

    # Save the image
    filename = f"fractal_depth_{depth}.png"
    image = Image.fromarray(image_array)
    image.save(filename)
    print(f"Saved {filename} (depth={depth})")


