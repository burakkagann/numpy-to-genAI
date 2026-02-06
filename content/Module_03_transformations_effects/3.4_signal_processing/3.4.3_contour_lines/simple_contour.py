"""
Quick Start: Simple Contour Visualization

Creates a basic stepped contour from a single Gaussian hill,
demonstrating how continuous height data becomes discrete bands.

Author: Claude (NumPy-to-GenAI Project)
Date: 2025-12-07

Thesis Metadata:
    Framework: F1 (Hands-On Discovery)
    Module: 3.4
    Exercise Type: Quick Start Demo
    Cognitive Load: Low
    Core Concepts: scalar fields, contour levels, Gaussian functions
    Prerequisites: Module 1.1.1 (Images as Arrays)

RQ Contributions:
    RQ1: Demonstrates visual-first pedagogy with immediate output
    RQ2: Low cognitive load introduction to scalar field concepts
    RQ5: Connects array operations to real-world terrain visualization

Learning Objectives:
    - See how 2D arrays represent height/elevation data
    - Understand stepped quantization for contour visualization
    - Create a simple terrain visualization from scratch
"""

import numpy as np
from PIL import Image

# Step 1: Create a coordinate grid (like a map)
size = 300
x = np.linspace(-5, 5, size)
y = np.linspace(-5, 5, size)
X, Y = np.meshgrid(x, y)

# Step 2: Create a single Gaussian "hill" centered at origin
height = np.exp(-(X**2 + Y**2) / 4)

# Step 3: Normalize to 0-1 range and quantize into 8 levels
normalized = (height - height.min()) / (height.max() - height.min())
contour = (normalized * 8).astype(np.uint8) * 32

# Step 4: Save the result
image = Image.fromarray(contour, mode='L')
image.save('simple_contour.png')
print("Done! Saved simple_contour.png - a stepped contour visualization")
