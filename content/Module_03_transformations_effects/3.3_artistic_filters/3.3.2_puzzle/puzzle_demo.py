"""
Puzzle Demo - Quick Start Example

This script demonstrates the basic concepts of array concatenation using
np.vstack (vertical stacking) and np.hstack (horizontal stacking) to
assemble puzzle pieces.

Author: Claude (NumPy-to-GenAI Project)
Date: 2025-01-20

Thesis Metadata:
    Framework: F1-Hands-On
    Module: Module_03_transformations_effects
    Exercise Type: Quick Start Demo
    Cognitive Load: Low
    New Concepts: vstack, hstack, array concatenation
    Prerequisites: NumPy array basics, image loading with PIL
"""

import numpy as np
from PIL import Image

# Step 1: Load two puzzle pieces
piece_a = np.array(Image.open('a.png'))
piece_b = np.array(Image.open('b.png'))

# Check the shapes - both should have the same height for hstack
print(f"Piece A shape: {piece_a.shape}")
print(f"Piece B shape: {piece_b.shape}")

# Step 2: Stack pieces horizontally (side by side)
# hstack requires arrays to have the SAME HEIGHT (first dimension)
top_row = np.hstack([piece_a, piece_b])
print(f"Top row shape after hstack: {top_row.shape}")

# Step 3: Save the result
Image.fromarray(top_row).save('quick_start_output.png')
print("\nSaved quick_start_output.png - pieces A and B combined horizontally!")
