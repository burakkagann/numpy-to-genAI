"""
Exercise 9.1.1 - Generative Art with Perceptrons

Uses multiple perceptrons as decision boundaries to partition a 2D canvas
into colored regions, creating geometric abstract art.

Inspired by:
  Rosenblatt, F. (1958). The Perceptron: A Probabilistic Model for
  Information Storage and Organization in the Brain.
  Psychological Review, 65(6), 386-408.
"""

import numpy as np
from PIL import Image

# Fixed seed for reproducible art
np.random.seed(42)


class Perceptron:
    def __init__(self, input_size):
        # Larger random weights create steeper decision boundaries,
        # producing more distinct angular regions on the canvas
        self.weights = np.random.randn(input_size) * 2
        self.bias = np.random.randn() * 0.5

    def forward(self, x):
        # Each perceptron divides the 2-D plane into two halves
        # along a line defined by  w1*x1 + w2*x2 + b = 0
        return 1 if np.dot(self.weights, x) + self.bias >= 0 else 0


# Create 4 perceptrons, each with a different random decision boundary
perceptrons = [Perceptron(2) for _ in range(4)]

# Build a 16-color palette (2^4 = 16 possible region combinations).
# The bit-manipulation formula spreads colors across the RGB cube
# so neighboring regions get visually distinct hues.
colors = [[80 + 140 * ((i * 23 % 16) % 4) // 3,
           80 + 140 * (((i * 23 % 16) // 2) % 4) // 3,
           80 + 140 * (((i * 23 % 16) // 4) % 4) // 3]
          for i in range(16)]

# Create a blank 400x400 RGB canvas
canvas = np.zeros((400, 400, 3), dtype=np.uint8)

# Assign a color to every pixel based on its perceptron "signature"
for y in range(400):
    for x in range(400):
        # Normalize pixel coordinates from [0, 400) to roughly [-2, 2]
        # so the decision boundaries cross near the center of the image
        point = np.array([(x - 200) / 100, (y - 200) / 100])

        # Each perceptron outputs 0 or 1 for this point.
        # Combine the four outputs into a 4-bit integer (0-15)
        # that uniquely identifies the region the pixel falls in.
        signature = sum(p.forward(point) * (2 ** i)
                       for i, p in enumerate(perceptrons))

        # Look up the color for this region
        canvas[y, x] = colors[signature]

Image.fromarray(canvas).save("perceptron_art.png")
print("Saved perceptron_art.png")
