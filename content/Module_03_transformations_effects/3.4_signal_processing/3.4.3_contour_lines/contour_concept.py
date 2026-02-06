"""
Concept Diagram: Contour Level Extraction

Creates a side-by-side visualization showing how continuous height data
is transformed into discrete contour levels. This diagram helps explain
the quantization process.

Author: Claude (NumPy-to-GenAI Project)
Date: 2025-12-07

Thesis Metadata:
    Framework: F1 (Hands-On Discovery)
    Module: 3.4 (Signal Processing)
    Exercise Type: Concept Diagram
    Cognitive Load: Low
    Core Concepts: quantization, scalar fields, visualization
"""

import numpy as np
from PIL import Image

# Create a 1D profile to show the concept clearly
# Left side: continuous gradient
# Right side: stepped/quantized version

height = 300
width = 600  # Wide image for side-by-side

# Create output array (grayscale)
output = np.zeros((height, width), dtype=np.uint8)

# Left half: continuous gradient (simulates smooth height data)
# Create a simple sinusoidal pattern for visual interest
for col in range(width // 2):
    for row in range(height):
        # Create a smooth gradient with some variation
        x_norm = col / (width // 2)  # 0 to 1
        y_norm = row / height  # 0 to 1

        # Smooth Gaussian-like value
        value = np.exp(-((x_norm - 0.5)**2 + (y_norm - 0.5)**2) / 0.15)
        output[row, col] = int(value * 255)

# Right half: same data but quantized into 8 levels
for col in range(width // 2, width):
    for row in range(height):
        # Same calculation
        x_norm = (col - width // 2) / (width // 2)
        y_norm = row / height

        value = np.exp(-((x_norm - 0.5)**2 + (y_norm - 0.5)**2) / 0.15)

        # Quantize to 8 levels (0-7), then scale to grayscale
        level = int(value * 8)
        if level > 7:
            level = 7
        output[row, col] = level * 32

# Add a vertical dividing line between the two halves
output[:, width // 2 - 1:width // 2 + 1] = 128

# Save the result
image = Image.fromarray(output, mode='L')
image.save('contour_concept.png')
print("Done! Saved contour_concept.png")
print("Left: Continuous scalar field (smooth gradients)")
print("Right: Quantized contour levels (8 discrete bands)")
