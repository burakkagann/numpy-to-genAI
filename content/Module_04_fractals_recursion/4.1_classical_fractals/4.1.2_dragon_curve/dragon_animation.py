"""
Exercise 4.1.2: Dragon Curve Animation

Create an animated GIF showing the dragon curve growing through
successive iterations (depths). This visualization helps learners
understand how fractal complexity emerges from simple recursive rules.
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


def move_forward(canvas, position, step_size, direction, color):
    """Draw a line segment in the current direction with bounds checking."""
    x, y = position
    h, w = canvas.shape[:2]

    if direction == RIGHT:
        end_x = min(x + step_size, w - 1)
        if 0 <= y < h and x < w:
            canvas[y, x:end_x] = color
        return (end_x, y)
    elif direction == LEFT:
        start_x = max(x - step_size, 0)
        if 0 <= y < h:
            canvas[y, start_x:x] = color
        return (start_x, y)
    elif direction == UP:
        start_y = max(y - step_size, 0)
        if 0 <= x < w:
            canvas[start_y:y, x] = color
        return (x, start_y)
    elif direction == DOWN:
        end_y = min(y + step_size, h - 1)
        if 0 <= x < w:
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


def add_depth_label(image_array, depth):
    """Add a depth label to the frame."""
    image = Image.fromarray(image_array)
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()

    label = f"Depth: {depth}"

    # Draw text with shadow for readability
    draw.text((11, 11), label, fill=(50, 50, 50), font=font)
    draw.text((10, 10), label, fill=(255, 255, 255), font=font)

    return np.array(image)


def create_frame(depth, canvas_size=400):
    """
    Create a single frame showing the dragon curve at the given depth.

    Parameters:
        depth: Recursion depth
        canvas_size: Size of the square canvas

    Returns:
        NumPy array with the frame
    """
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

    # Calculate step size to fit the dragon curve in the canvas
    # Lower depths use larger steps, higher depths use smaller steps
    step_size = max(1, int(canvas_size / (2 ** (depth / 2 + 2))))

    # Starting position - centered to fit the curve
    start_x = int(canvas_size * 0.68)
    start_y = int(canvas_size * 0.32)

    # Generate a color gradient based on depth (blue to purple)
    blue = int(255 - depth * 10)
    red = int(50 + depth * 15)
    color = (red, 100, blue)

    # Generate and draw the dragon curve
    sequence = generate_dragon_sequence('R', depth)
    draw_dragon_curve(canvas, sequence, (start_x, start_y), step_size, UP, color)

    # Add depth label
    canvas = add_depth_label(canvas, depth)

    return canvas


# =============================================================================
# Main Script
# =============================================================================

if __name__ == "__main__":
    # Animation parameters
    max_depth = 12
    canvas_size = 400

    frames = []

    print("Generating animation frames...")

    # Generate frames for each depth
    for depth in range(1, max_depth + 1):
        print(f"  Frame {depth}/{max_depth} (depth {depth})...")
        frame = create_frame(depth, canvas_size)
        frames.append(frame)

        # Hold the last frame longer (3 copies)
        if depth == max_depth:
            frames.extend([frame, frame, frame])

    # Also add copies of middle frames for pacing
    # Hold depth 6 and 8 an extra frame
    frames_with_pauses = []
    for i, frame in enumerate(frames[:-3]):  # Exclude the extra last frames
        frames_with_pauses.append(frame)
        depth = i + 1
        if depth in [6, 8, 10]:
            frames_with_pauses.append(frame)  # Extra frame for these depths

    # Add the held last frames
    frames_with_pauses.extend(frames[-3:])

    # Save as animated GIF
    print("Saving animation...")
    imageio.mimsave(
        'dragon_animation.gif',
        frames_with_pauses,
        fps=2,  # Slow enough to observe each depth
        loop=0  # Infinite loop
    )

    print(f"Animation saved as dragon_animation.gif")
    print(f"Total frames: {len(frames_with_pauses)}")
    print(f"Frame size: {canvas_size}x{canvas_size} pixels")
