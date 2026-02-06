import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Create a figure showing the kaleidoscope construction process
# Three panels: Original Wedge | Reflected Wedge | Combined Pattern

panel_size = 300
spacing = 20
total_width = panel_size * 3 + spacing * 4
total_height = panel_size + spacing * 2 + 40  # Extra space for labels

# Create the canvas
canvas = np.ones((total_height, total_width, 3), dtype=np.uint8) * 240  # Light gray background

def create_wedge_pattern(size, wedge_angle_deg=60, fill_wedge=True):
    """Create a pattern showing a wedge slice."""
    center = size // 2
    pattern = np.ones((size, size, 3), dtype=np.uint8) * 255  # White background

    # Create coordinate grids
    y, x = np.ogrid[:size, :size]
    x_centered = x - center
    y_centered = y - center

    # Calculate polar coordinates
    angle = np.arctan2(y_centered, x_centered)
    radius = np.sqrt(x_centered**2 + y_centered**2)

    # Convert wedge angle to radians
    wedge_angle = np.deg2rad(wedge_angle_deg)

    # Create colorful pattern within the wedge
    if fill_wedge:
        # Color gradient based on angle and radius within wedge
        in_wedge = (angle >= 0) & (angle < wedge_angle) & (radius < center - 10)

        # Create color gradient
        red = ((np.sin(angle * 5 + radius * 0.1) + 1) * 100 + 50).astype(np.uint8)
        green = ((np.cos(angle * 3) + 1) * 80 + 100).astype(np.uint8)
        blue = ((np.sin(radius * 0.15) + 1) * 100 + 50).astype(np.uint8)

        pattern[in_wedge, 0] = red[in_wedge]
        pattern[in_wedge, 1] = green[in_wedge]
        pattern[in_wedge, 2] = blue[in_wedge]

    # Draw wedge boundary lines
    # Line at angle = 0 (horizontal right)
    for r in range(center - 10):
        px = center + r
        py = center
        if 0 <= px < size and 0 <= py < size:
            pattern[py, px] = [0, 0, 0]  # Black line

    # Line at angle = wedge_angle
    for r in range(center - 10):
        px = int(center + r * np.cos(wedge_angle))
        py = int(center + r * np.sin(wedge_angle))
        if 0 <= px < size and 0 <= py < size:
            pattern[py, px] = [0, 0, 0]

    # Draw arc at the edge
    for a in np.linspace(0, wedge_angle, 50):
        px = int(center + (center - 15) * np.cos(a))
        py = int(center + (center - 15) * np.sin(a))
        if 0 <= px < size and 0 <= py < size:
            pattern[py, px] = [0, 0, 0]

    # Draw center point
    pattern[center-2:center+2, center-2:center+2] = [255, 0, 0]  # Red center dot

    return pattern

def create_reflected_wedge(size, wedge_angle_deg=60):
    """Create the wedge with its reflection (forming a kite shape)."""
    center = size // 2
    pattern = np.ones((size, size, 3), dtype=np.uint8) * 255

    y, x = np.ogrid[:size, :size]
    x_centered = x - center
    y_centered = y - center

    angle = np.arctan2(y_centered, x_centered)
    radius = np.sqrt(x_centered**2 + y_centered**2)

    wedge_angle = np.deg2rad(wedge_angle_deg)

    # Original wedge (0 to wedge_angle)
    in_original = (angle >= 0) & (angle < wedge_angle) & (radius < center - 10)

    # Reflected wedge (0 to -wedge_angle, which is 2*pi - wedge_angle to 2*pi)
    in_reflected = (angle < 0) & (angle >= -wedge_angle) & (radius < center - 10)

    # Color the original wedge
    red = ((np.sin(angle * 5 + radius * 0.1) + 1) * 100 + 50).astype(np.uint8)
    green = ((np.cos(angle * 3) + 1) * 80 + 100).astype(np.uint8)
    blue = ((np.sin(radius * 0.15) + 1) * 100 + 50).astype(np.uint8)

    pattern[in_original, 0] = red[in_original]
    pattern[in_original, 1] = green[in_original]
    pattern[in_original, 2] = blue[in_original]

    # Color the reflected wedge (mirror the angle for color calculation)
    angle_mirrored = -angle  # Mirror angle for color sampling
    red_mirror = ((np.sin(angle_mirrored * 5 + radius * 0.1) + 1) * 100 + 50).astype(np.uint8)
    green_mirror = ((np.cos(angle_mirrored * 3) + 1) * 80 + 100).astype(np.uint8)
    blue_mirror = ((np.sin(radius * 0.15) + 1) * 100 + 50).astype(np.uint8)

    pattern[in_reflected, 0] = red_mirror[in_reflected]
    pattern[in_reflected, 1] = green_mirror[in_reflected]
    pattern[in_reflected, 2] = blue_mirror[in_reflected]

    # Draw boundary lines
    for r in range(center - 10):
        # Upper edge
        px = int(center + r * np.cos(wedge_angle))
        py = int(center + r * np.sin(wedge_angle))
        if 0 <= px < size and 0 <= py < size:
            pattern[py, px] = [0, 0, 0]
        # Lower edge (reflected)
        py_r = int(center - r * np.sin(wedge_angle))
        if 0 <= px < size and 0 <= py_r < size:
            pattern[py_r, px] = [0, 0, 0]

    # Draw center point
    pattern[center-2:center+2, center-2:center+2] = [255, 0, 0]

    return pattern

def create_full_kaleidoscope(size, num_folds=6):
    """Create the full kaleidoscope pattern from wedge replication."""
    center = size // 2
    pattern = np.ones((size, size, 3), dtype=np.uint8) * 255

    y, x = np.ogrid[:size, :size]
    x_centered = x - center
    y_centered = y - center

    angle = np.arctan2(y_centered, x_centered)
    radius = np.sqrt(x_centered**2 + y_centered**2)

    wedge_angle = 2 * np.pi / num_folds

    # Map all angles to a single wedge with reflection
    angle_positive = angle + np.pi
    wedge_index = (angle_positive / wedge_angle).astype(int)
    angle_in_wedge = angle_positive - wedge_index * wedge_angle

    # Mirror odd-numbered wedges
    is_odd = wedge_index % 2 == 1
    angle_mirrored = np.where(is_odd, wedge_angle - angle_in_wedge, angle_in_wedge)

    # Create colors based on mirrored angle
    in_circle = radius < center - 10

    red = ((np.sin(angle_mirrored * 5 + radius * 0.1) + 1) * 100 + 50).astype(np.uint8)
    green = ((np.cos(angle_mirrored * 3) + 1) * 80 + 100).astype(np.uint8)
    blue = ((np.sin(radius * 0.15) + 1) * 100 + 50).astype(np.uint8)

    pattern[in_circle, 0] = red[in_circle]
    pattern[in_circle, 1] = green[in_circle]
    pattern[in_circle, 2] = blue[in_circle]

    # Draw radial lines at wedge boundaries
    for i in range(num_folds):
        boundary_angle = i * wedge_angle - np.pi
        for r in range(center - 10):
            px = int(center + r * np.cos(boundary_angle))
            py = int(center + r * np.sin(boundary_angle))
            if 0 <= px < size and 0 <= py < size:
                pattern[py, px] = [50, 50, 50]  # Dark gray lines

    # Draw center point
    pattern[center-2:center+2, center-2:center+2] = [255, 0, 0]

    return pattern

# Generate the three panels
wedge = create_wedge_pattern(panel_size)
reflected = create_reflected_wedge(panel_size)
full = create_full_kaleidoscope(panel_size)

# Place panels on canvas
x_offset = spacing
canvas[spacing:spacing+panel_size, x_offset:x_offset+panel_size] = wedge

x_offset = spacing * 2 + panel_size
canvas[spacing:spacing+panel_size, x_offset:x_offset+panel_size] = reflected

x_offset = spacing * 3 + panel_size * 2
canvas[spacing:spacing+panel_size, x_offset:x_offset+panel_size] = full

# Add labels using PIL
output_image = Image.fromarray(canvas)
draw = ImageDraw.Draw(output_image)

# Add labels below each panel
label_y = spacing + panel_size + 5
labels = ["1. Single Wedge (60°)", "2. Wedge + Reflection", "3. 6-Fold Kaleidoscope"]
x_positions = [spacing + panel_size//2, spacing*2 + panel_size + panel_size//2, spacing*3 + panel_size*2 + panel_size//2]

for label, x_pos in zip(labels, x_positions):
    # Get text bounding box for centering
    bbox = draw.textbbox((0, 0), label)
    text_width = bbox[2] - bbox[0]
    draw.text((x_pos - text_width//2, label_y), label, fill=(0, 0, 0))

# Add arrows between panels
arrow_y = spacing + panel_size // 2
arrow_x1 = spacing + panel_size + 5
arrow_x2 = spacing * 2 + panel_size - 5
draw.text((arrow_x1 + (arrow_x2 - arrow_x1)//2 - 10, arrow_y - 5), "->", fill=(100, 100, 100))

arrow_x1 = spacing * 2 + panel_size * 2 + 5
arrow_x2 = spacing * 3 + panel_size * 2 - 5
draw.text((arrow_x1 + (arrow_x2 - arrow_x1)//2 - 10, arrow_y - 5), "->", fill=(100, 100, 100))

# Save the diagram
output_image.save('wedge_reflection.png')
