"""
Exercise 4.1.3: Complex Plane and Iteration Visualization

Educational diagram showing:
1. The complex plane with real and imaginary axes
2. The escape boundary (|z| = 2 circle)
3. Sample iteration trajectories for points inside and outside the Mandelbrot set

This helps students visualize what the Mandelbrot algorithm is actually computing.

Author: Claude (NumPy-to-GenAI Project)
Date: 2025-01-22
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def iterate_mandelbrot(c, max_iterations):
    """
    Perform Mandelbrot iteration for a single point c.
    Returns list of z values showing the trajectory.
    """
    z = 0
    trajectory = [z]

    for _ in range(max_iterations):
        z = z * z + c
        trajectory.append(z)
        if abs(z) > 2:  # Escaped
            break

    return trajectory


def complex_to_pixel(z, center, scale, width, height):
    """Convert complex number to pixel coordinates."""
    x = int(width / 2 + (z.real - center.real) * scale)
    y = int(height / 2 - (z.imag - center.imag) * scale)  # Flip y-axis
    return (x, y)


# =============================================================================
# Create educational diagram
# =============================================================================

print("Creating complex plane diagram...")

# Image parameters
width, height = 600, 600
center = complex(-0.5, 0)  # Center the view
scale = 120  # Pixels per unit

# Create white background
image = Image.new('RGB', (width, height), 'white')
draw = ImageDraw.Draw(image)

# Draw grid lines
grid_color = (220, 220, 220)
for i in range(-4, 5):
    # Vertical lines (real axis values)
    x = width // 2 + i * scale
    draw.line([(x, 0), (x, height)], fill=grid_color, width=1)
    # Horizontal lines (imaginary axis values)
    y = height // 2 - i * scale
    draw.line([(0, y), (width, y)], fill=grid_color, width=1)

# Draw axes (thicker)
axis_color = (100, 100, 100)
# Real axis
draw.line([(0, height // 2), (width, height // 2)], fill=axis_color, width=2)
# Imaginary axis
draw.line([(width // 2, 0), (width // 2, height)], fill=axis_color, width=2)

# Draw escape circle |z| = 2
circle_color = (200, 50, 50)
circle_radius = 2 * scale
circle_center = complex_to_pixel(0j, center, scale, width, height)
cx, cy = circle_center
draw.ellipse(
    [(cx - circle_radius, cy - circle_radius),
     (cx + circle_radius, cy + circle_radius)],
    outline=circle_color, width=3
)

# Label the escape circle
draw.text((cx + circle_radius + 10, cy - 10), "|z| = 2", fill=circle_color)
draw.text((cx + circle_radius + 10, cy + 5), "(escape boundary)", fill=circle_color)

# Draw axis labels
label_color = (50, 50, 50)
draw.text((width - 50, height // 2 + 10), "Real", fill=label_color)
draw.text((width // 2 + 10, 10), "Imaginary", fill=label_color)

# Mark some points on axes
for i in range(-2, 3):
    if i == 0:
        continue
    # Real axis marks
    x = width // 2 + i * scale
    draw.line([(x, height // 2 - 5), (x, height // 2 + 5)], fill=axis_color, width=2)
    draw.text((x - 10, height // 2 + 10), str(i), fill=label_color)
    # Imaginary axis marks
    y = height // 2 - i * scale
    draw.line([(width // 2 - 5, y), (width // 2 + 5, y)], fill=axis_color, width=2)
    draw.text((width // 2 + 10, y - 5), f"{i}i", fill=label_color)

# =============================================================================
# Draw example trajectories
# =============================================================================

# Example 1: Point INSIDE the set (doesn't escape)
c_inside = complex(-0.1, 0.7)  # A point in the Mandelbrot set
trajectory_inside = iterate_mandelbrot(c_inside, 20)
inside_color = (0, 150, 0)

# Draw trajectory
points_inside = [complex_to_pixel(z, center, scale, width, height) for z in trajectory_inside]
for i in range(len(points_inside) - 1):
    draw.line([points_inside[i], points_inside[i + 1]], fill=inside_color, width=2)

# Mark the starting point c
start_inside = complex_to_pixel(c_inside, center, scale, width, height)
draw.ellipse(
    [(start_inside[0] - 6, start_inside[1] - 6),
     (start_inside[0] + 6, start_inside[1] + 6)],
    fill=inside_color
)
draw.text((start_inside[0] + 10, start_inside[1] - 15),
          f"c = {c_inside.real}+{c_inside.imag}i", fill=inside_color)
draw.text((start_inside[0] + 10, start_inside[1]),
          "(bounded, IN set)", fill=inside_color)

# Example 2: Point OUTSIDE the set (escapes quickly)
c_outside = complex(0.5, 0.5)  # A point outside the Mandelbrot set
trajectory_outside = iterate_mandelbrot(c_outside, 10)
outside_color = (200, 50, 50)

# Draw trajectory
points_outside = [complex_to_pixel(z, center, scale, width, height) for z in trajectory_outside]
for i in range(len(points_outside) - 1):
    p1, p2 = points_outside[i], points_outside[i + 1]
    # Only draw if both points are in frame
    if 0 <= p1[0] < width and 0 <= p1[1] < height:
        draw.line([p1, p2], fill=outside_color, width=2)

# Mark the starting point c
start_outside = complex_to_pixel(c_outside, center, scale, width, height)
draw.ellipse(
    [(start_outside[0] - 6, start_outside[1] - 6),
     (start_outside[0] + 6, start_outside[1] + 6)],
    fill=outside_color
)
draw.text((start_outside[0] + 10, start_outside[1] - 15),
          f"c = {c_outside.real}+{c_outside.imag}i", fill=outside_color)
draw.text((start_outside[0] + 10, start_outside[1]),
          "(escapes, NOT in set)", fill=outside_color)

# Add legend/explanation at bottom
legend_y = height - 80
draw.rectangle([(10, legend_y), (width - 10, height - 10)], fill=(245, 245, 245), outline=(200, 200, 200))
draw.text((20, legend_y + 10),
          "Mandelbrot Iteration: z(n+1) = z(n)^2 + c, starting with z(0) = 0",
          fill=(50, 50, 50))
draw.text((20, legend_y + 30),
          "Green trajectory: Point stays bounded (IN the Mandelbrot set)",
          fill=inside_color)
draw.text((20, legend_y + 50),
          "Red trajectory: Point escapes past |z|=2 (NOT in the set)",
          fill=outside_color)

# Add title
draw.rectangle([(10, 10), (width - 10, 50)], fill=(245, 245, 245), outline=(200, 200, 200))
draw.text((20, 20), "Complex Plane: Mandelbrot Iteration Visualization", fill=(50, 50, 100))

# Save diagram
image.save('complex_plane_diagram.png')
print("Complex plane diagram saved as 'complex_plane_diagram.png'")
print("\nThis diagram shows:")
print("  - The complex plane with real (horizontal) and imaginary (vertical) axes")
print("  - The escape boundary circle where |z| = 2")
print("  - Two example trajectories: one that stays bounded, one that escapes")
