"""
Exercise 2.1.1: Sunburst Pattern (Exercise 3 Solution)

Creates a radial "sunburst" pattern with lines emanating from the center
to evenly-spaced points on the canvas edge. This exercise combines
line drawing with trigonometry to create symmetrical generative art.

Author: Claude (NumPy-to-GenAI Project)
Date: 2025-01-30

Thesis Metadata:
    Framework: F1 (Hands-On Discovery)
    Module: Module_02_geometry_mathematics
    Exercise Type: Re-code (Exercise 3)
    Cognitive Load: MEDIUM
    New Concepts: Polar coordinates, trigonometry in generative art
    Prerequisites: Line drawing, basic trigonometry

Research Question Contributions:
    RQ1 (Framework Design): Re-code exercise requires synthesis
    RQ2 (Cognitive Load): Scaffolded approach reduces complexity
    RQ4 (Assessment): Creative dimension - design aesthetic patterns
    RQ5 (Transfer): Radial thinking transfers to spirals, mandalas
"""

import numpy as np
from PIL import Image


def draw_line(canvas, x_start, y_start, x_end, y_end):
    """Draw a line from start to end point using linear interpolation."""
    num_points = max(abs(x_end - x_start) + 1, abs(y_end - y_start) + 1)
    x_coords = np.linspace(x_start, x_end, num_points).round().astype(int)
    y_coords = np.linspace(y_start, y_end, num_points).round().astype(int)
    canvas[y_coords, x_coords] = 255


# Step 1: Create blank canvas
canvas = np.zeros((400, 400), dtype=np.uint8)

# Step 2: Define center point
center_x, center_y = 200, 200

# Step 3: Define number of rays and radius
num_rays = 24  # More rays = more detailed sunburst
radius = 180   # Distance from center to edge of rays

# Step 4: Calculate angles for evenly-spaced rays
# We want angles from 0 to 2π (full circle)
angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)

# Step 5: Draw rays using polar-to-Cartesian conversion
for angle in angles:
    # Convert polar coordinates (angle, radius) to Cartesian (x, y)
    # x = r * cos(θ), y = r * sin(θ)
    end_x = int(center_x + radius * np.cos(angle))
    end_y = int(center_y + radius * np.sin(angle))

    # Draw line from center to calculated endpoint
    draw_line(canvas, center_x, center_y, end_x, end_y)

print("Sunburst pattern saved as sunburst_example.png")
print(f"Created {num_rays} rays radiating from ({center_x}, {center_y})")
print(f"Radius: {radius} pixels")

# Step 6: Save result
output_image = Image.fromarray(canvas)
output_image.save('sunburst_example.png')
