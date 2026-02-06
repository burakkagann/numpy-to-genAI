"""
Warhol Concept Diagram - RGB Channel Rotation Visualization

Creates a visual diagram explaining how channel rotation transforms colors.
Shows the six possible permutations of RGB channels.

Framework: F1 (Hands-On Discovery)
Concepts: RGB channel permutations, color theory visualization
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Create a sample color bar to demonstrate channel effects
bar_height = 80
bar_width = 120
sample = np.zeros((bar_height, bar_width, 3), dtype=np.uint8)

# Create a gradient with distinct RGB regions
for x in range(bar_width):
    # Transition through colors: Red -> Yellow -> Green -> Cyan -> Blue -> Magenta
    position = x / bar_width * 6
    if position < 1:  # Red to Yellow
        sample[:, x] = [255, int(255 * position), 0]
    elif position < 2:  # Yellow to Green
        sample[:, x] = [int(255 * (2 - position)), 255, 0]
    elif position < 3:  # Green to Cyan
        sample[:, x] = [0, 255, int(255 * (position - 2))]
    elif position < 4:  # Cyan to Blue
        sample[:, x] = [0, int(255 * (4 - position)), 255]
    elif position < 5:  # Blue to Magenta
        sample[:, x] = [int(255 * (position - 4)), 0, 255]
    else:  # Magenta to Red
        sample[:, x] = [255, 0, int(255 * (6 - position))]

# Define all 6 channel permutations
permutations = [
    ([0, 1, 2], "Original [R,G,B]"),
    ([0, 2, 1], "[R,B,G]"),
    ([1, 0, 2], "[G,R,B]"),
    ([1, 2, 0], "[G,B,R]"),
    ([2, 0, 1], "[B,R,G]"),
    ([2, 1, 0], "[B,G,R]"),
]

# Create the diagram canvas
padding = 20
label_height = 30
cols = 3
rows = 2
canvas_width = cols * bar_width + (cols + 1) * padding
canvas_height = rows * (bar_height + label_height) + (rows + 1) * padding + 60

canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 240

# Add title area
title_y = 15

# Place each permutation
for idx, (perm, label) in enumerate(permutations):
    row = idx // cols
    col = idx % cols

    x_start = padding + col * (bar_width + padding)
    y_start = 60 + padding + row * (bar_height + label_height + padding)

    # Apply the channel permutation
    transformed = sample[:, :, perm]

    # Place on canvas
    canvas[y_start:y_start + bar_height, x_start:x_start + bar_width] = transformed

# Convert to PIL for adding text
output = Image.fromarray(canvas, mode='RGB')
draw = ImageDraw.Draw(output)

# Try to use a basic font
try:
    font = ImageFont.truetype("arial.ttf", 14)
    title_font = ImageFont.truetype("arial.ttf", 18)
except:
    font = ImageFont.load_default()
    title_font = font

# Add title
draw.text((canvas_width // 2 - 120, 15), "RGB Channel Permutations", fill=(30, 30, 30), font=title_font)

# Add labels under each bar
for idx, (perm, label) in enumerate(permutations):
    row = idx // cols
    col = idx % cols

    x_start = padding + col * (bar_width + padding)
    y_start = 60 + padding + row * (bar_height + label_height + padding)

    # Draw label centered under the bar
    label_x = x_start + bar_width // 2 - len(label) * 3
    label_y = y_start + bar_height + 5
    draw.text((label_x, label_y), label, fill=(50, 50, 50), font=font)

output.save('channel_rotation_diagram.png')
print("Saved channel_rotation_diagram.png - Shows all 6 RGB permutations!")
