"""
Generate an animated GIF showing the twin dragon curves,
two interlocking dragon curves that tile together.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio

# Direction constants
LEFT, UP, RIGHT, DOWN = range(4)


def turn_left(direction):
    """Rotate 90 degrees counter-clockwise."""
    return (direction + 3) % 4


def turn_right(direction):
    """Rotate 90 degrees clockwise."""
    return (direction + 1) % 4


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


def draw_dragon_curve(canvas, sequence, start_position, step_size, start_direction, color):
    """Draw the dragon curve on a canvas."""
    position = start_position
    direction = start_direction
    h, w = canvas.shape[:2]

    for turn in sequence:
        x, y = position

        # Draw line segment with bounds checking
        if direction == RIGHT:
            end_x = min(x + step_size, w - 1)
            if 0 <= y < h and x < w:
                canvas[y, x:end_x] = color
            position = (end_x, y)
        elif direction == LEFT:
            start_x = max(x - step_size, 0)
            if 0 <= y < h:
                canvas[y, start_x:x] = color
            position = (start_x, y)
        elif direction == UP:
            start_y = max(y - step_size, 0)
            if 0 <= x < w:
                canvas[start_y:y, x] = color
            position = (x, start_y)
        elif direction == DOWN:
            end_y = min(y + step_size, h - 1)
            if 0 <= x < w:
                canvas[y:end_y, x] = color
            position = (x, end_y)

        # Turn
        if turn == 'R':
            direction = turn_right(direction)
        else:
            direction = turn_left(direction)

    # Final segment
    x, y = position
    if direction == RIGHT:
        end_x = min(x + step_size, w - 1)
        if 0 <= y < h and x < w:
            canvas[y, x:end_x] = color
    elif direction == LEFT:
        start_x = max(x - step_size, 0)
        if 0 <= y < h:
            canvas[y, start_x:x] = color
    elif direction == UP:
        start_y = max(y - step_size, 0)
        if 0 <= x < w:
            canvas[start_y:y, x] = color
    elif direction == DOWN:
        end_y = min(y + step_size, h - 1)
        if 0 <= x < w:
            canvas[y:end_y, x] = color


def add_label(image_array, text):
    """Add a text label to the frame."""
    image = Image.fromarray(image_array)
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    draw.text((11, 11), text, fill=(50, 50, 50), font=font)
    draw.text((10, 10), text, fill=(255, 255, 255), font=font)

    return np.array(image)


if __name__ == "__main__":
    canvas_size = 400
    max_depth = 10
    step_size = 3

    # Starting position (center)
    start_x = int(canvas_size * 0.5)
    start_y = int(canvas_size * 0.5)

    frames = []

    # Colors for the two dragons
    blue_color = (100, 180, 255)
    orange_color = (255, 150, 100)

    print("Generating twin dragon animation...")

    for depth in range(1, max_depth + 1):
        print(f"  Frame {depth}/{max_depth}...")

        canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
        canvas[:] = 20  # Dark background

        sequence = generate_dragon_sequence('R', depth)

        # Adjust step size for each depth
        current_step = max(1, int(canvas_size / (2 ** (depth / 2 + 2))))

        # Adjust starting position for each depth
        offset_x = int(canvas_size * 0.55)
        offset_y = int(canvas_size * 0.45)

        # Draw first dragon (blue, facing UP)
        draw_dragon_curve(canvas, sequence, (offset_x, offset_y),
                          current_step, UP, blue_color)

        # Draw second dragon (orange, facing DOWN)
        draw_dragon_curve(canvas, sequence, (offset_x, offset_y),
                          current_step, DOWN, orange_color)

        # Add label
        canvas = add_label(canvas, f"Twin Dragon - Depth {depth}")
        frames.append(canvas)

        # Hold certain frames longer
        if depth in [6, 8, 10]:
            frames.append(canvas)

        # Hold final frame extra long
        if depth == max_depth:
            for _ in range(4):
                frames.append(canvas)

    # Save as animated GIF
    imageio.mimsave(
        'twin_dragon.gif',
        frames,
        fps=2,
        loop=0
    )

    print(f"Saved twin_dragon.gif ({len(frames)} frames)")
