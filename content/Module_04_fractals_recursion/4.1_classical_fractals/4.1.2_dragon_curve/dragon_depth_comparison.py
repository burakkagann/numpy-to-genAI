"""
Exercise 4.1.2: Dragon Curve Depth Comparison

Generate a comparison grid showing the dragon curve at different
recursion depths (iterations). This visualization demonstrates how
the fractal emerges from simple rules applied repeatedly.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Direction constants
LEFT, UP, RIGHT, DOWN = range(4)


def turn_left(direction):
    """Rotate 90 degrees counter-clockwise."""
    return (direction + 3) % 4


def turn_right(direction):
    """Rotate 90 degrees clockwise."""
    return (direction + 1) % 4


def move_forward(canvas, position, step_size, direction, color):
    """Draw a line segment in the current direction."""
    x, y = position
    h, w = canvas.shape[:2]

    if direction == RIGHT:
        end_x = min(x + step_size, w - 1)
        canvas[y, x:end_x] = color
        return (end_x, y)
    elif direction == LEFT:
        start_x = max(x - step_size, 0)
        canvas[y, start_x:x] = color
        return (start_x, y)
    elif direction == UP:
        start_y = max(y - step_size, 0)
        canvas[start_y:y, x] = color
        return (x, start_y)
    elif direction == DOWN:
        end_y = min(y + step_size, h - 1)
        canvas[y:end_y, x] = color
        return (x, end_y)


def invert_sequence(sequence):
    """Swap all L's and R's in the sequence."""
    return ''.join(['L' if char == 'R' else 'R' for char in sequence])


def generate_dragon_sequence(initial_turn, depth):
    """Generate the dragon curve turn sequence recursively."""
    if depth == 0:
        return initial_turn
    else:
        previous = generate_dragon_sequence(initial_turn, depth - 1)
        second_half = invert_sequence(previous[::-1])
        return previous + 'R' + second_half


def draw_dragon_curve(canvas, sequence, start_position, step_size,
                      start_direction, color):
    """Render the dragon curve using turtle graphics."""
    position = start_position
    direction = start_direction

    for turn in sequence:
        position = move_forward(canvas, position, step_size, direction, color)
        if turn == 'R':
            direction = turn_right(direction)
        else:
            direction = turn_left(direction)
    position = move_forward(canvas, position, step_size, direction, color)


def create_dragon_at_depth(depth, cell_size=250, step_size=None, color=(100, 180, 255)):
    """
    Create a single dragon curve image at the specified depth.

    Parameters:
        depth: Recursion depth for the dragon curve
        cell_size: Size of the output image (square)
        step_size: Pixel size per step (auto-calculated if None)
        color: RGB color tuple

    Returns:
        NumPy array with the dragon curve image
    """
    canvas = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)

    # Auto-calculate step size based on depth to fit in cell
    if step_size is None:
        # Larger step for lower depths, smaller for higher
        step_size = max(1, int(cell_size / (2 ** (depth / 2 + 2))))

    # Starting position adjusted for each depth to center the curve
    # Higher depths need to start more to the right and higher up
    start_x = int(cell_size * 0.65)
    start_y = int(cell_size * 0.35)

    sequence = generate_dragon_sequence('R', depth)
    draw_dragon_curve(canvas, sequence, (start_x, start_y), step_size, UP, color)

    return canvas


def add_label(image_array, label, position='top'):
    """Add a text label to an image."""
    image = Image.fromarray(image_array)
    draw = ImageDraw.Draw(image)

    # Try to use a system font, fall back to default
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    # Position the text
    if position == 'top':
        text_position = (10, 5)
    else:
        text_position = (10, image_array.shape[0] - 25)

    # Draw text with a slight shadow for readability
    draw.text((text_position[0] + 1, text_position[1] + 1), label, fill=(50, 50, 50), font=font)
    draw.text(text_position, label, fill=(255, 255, 255), font=font)

    return np.array(image)


# =============================================================================
# Main Script
# =============================================================================

if __name__ == "__main__":
    # Depths to compare
    depths = [2, 4, 6, 8, 10, 12]

    # Grid layout: 2 rows x 3 columns
    rows, cols = 2, 3
    cell_size = 280
    padding = 10

    # Calculate total image size
    total_width = cols * cell_size + (cols + 1) * padding
    total_height = rows * cell_size + (rows + 1) * padding

    # Create the comparison grid with dark gray background
    grid = np.ones((total_height, total_width, 3), dtype=np.uint8) * 30

    # Generate and place each dragon curve
    for idx, depth in enumerate(depths):
        row = idx // cols
        col = idx % cols

        # Create the dragon curve at this depth
        print(f"Generating depth {depth}...")
        dragon_image = create_dragon_at_depth(depth, cell_size)

        # Add depth label
        dragon_image = add_label(dragon_image, f"Depth {depth}")

        # Calculate position in the grid
        x_start = padding + col * (cell_size + padding)
        y_start = padding + row * (cell_size + padding)

        # Place the image in the grid
        grid[y_start:y_start + cell_size, x_start:x_start + cell_size] = dragon_image

    # Save the comparison grid
    result = Image.fromarray(grid)
    result.save('dragon_depth_comparison.png')
    print(f"Comparison grid saved as dragon_depth_comparison.png")
    print(f"Grid size: {total_width}x{total_height} pixels")
