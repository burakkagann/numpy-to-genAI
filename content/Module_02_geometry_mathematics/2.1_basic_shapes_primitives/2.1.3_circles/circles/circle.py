"""
Exercise 2.1.3: Drawing Circles with Distance Calculations

This exercise teaches how to render circles mathematically using the
Euclidean distance formula. Instead of drawing pixel-by-pixel, we
calculate which pixels fall inside the circle using vectorized NumPy
operations - a fundamental technique for generative art.

Author: Claude (NumPy-to-GenAI Project)
Date: 2025-01-30

Thesis Metadata:
    Framework: F1-Hands-On
    Module: Module_02_geometry_mathematics
    Exercise Type: Execute
    Cognitive Load: Low
    New Concepts: Distance formula, np.ogrid coordinate grids, boolean masking
    Prerequisites: Module 1.1.1 (RGB Basics), Module 2.1.1 (Lines)

Research Question Contributions:
    RQ1 (Framework Design): Demonstrates mathematical-to-visual pattern
    RQ2 (Cognitive Load): Simple script introduces one core algorithm
    RQ4 (Assessment): Technical accuracy + creative parameter exploration
    RQ5 (Transfer): Masking technique applies to ellipses, rings, gradients

Learning Objectives:
    - Understand how circles are defined by distance from center
    - Use np.ogrid to create coordinate grids efficiently
    - Apply boolean masking to select pixels inside a shape
"""

import numpy as np
from PIL import Image

# =============================================================================
# Configuration: Circle parameters (try changing these!)
# =============================================================================
CANVAS_SIZE = 512           # Width and height of the output image
CENTER_X = 256              # X coordinate of circle center
CENTER_Y = 256              # Y coordinate of circle center
RADIUS = 150                # Circle radius in pixels
CIRCLE_COLOR = [255, 128, 0]  # Orange color (RGB)

# =============================================================================
# Step 1: Create coordinate grids using np.ogrid
# =============================================================================
# np.ogrid creates two arrays: Y contains row indices, X contains column indices
# This allows us to calculate distances for ALL pixels at once (vectorized)
Y, X = np.ogrid[0:CANVAS_SIZE, 0:CANVAS_SIZE]

# =============================================================================
# Step 2: Calculate squared distance from each pixel to the center
# =============================================================================
# Using the Pythagorean theorem: distance² = (x - cx)² + (y - cy)²
# We use squared distance to avoid the costly sqrt() operation
square_distance = (X - CENTER_X) ** 2 + (Y - CENTER_Y) ** 2

# =============================================================================
# Step 3: Create a boolean mask for pixels inside the circle
# =============================================================================
# A pixel is inside the circle if its distance < radius
# Comparing squared values: distance² < radius² is equivalent to distance < radius
inside_circle = square_distance < RADIUS ** 2

# =============================================================================
# Step 4: Create canvas and apply the mask
# =============================================================================
# Start with a black canvas (all zeros)
canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8)

# Apply color only to pixels where the mask is True
canvas[inside_circle] = CIRCLE_COLOR

# =============================================================================
# Step 5: Save the result
# =============================================================================
output_image = Image.fromarray(canvas, mode='RGB')
output_image.save('circle.png')

# Provide feedback to the user
print("Circle created successfully!")
print(f"  Center: ({CENTER_X}, {CENTER_Y})")
print(f"  Radius: {RADIUS} pixels")
print(f"  Color: RGB{tuple(CIRCLE_COLOR)}")
print(f"  Canvas size: {CANVAS_SIZE}x{CANVAS_SIZE}")
print(f"  Output saved as: circle.png")
