import numpy as np
from PIL import Image

np.random.seed(42)


class Perceptron:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size) * 2
        self.bias = np.random.randn() * 0.5

    def forward(self, x):
        return 1 if np.dot(self.weights, x) + self.bias >= 0 else 0


# Create 4 perceptrons with different random boundaries
perceptrons = [Perceptron(2) for _ in range(4)]

# Generate color palette (16 colors for 2^4 regions)
colors = [[80 + 140 * ((i * 23 % 16) % 4) // 3,
           80 + 140 * (((i * 23 % 16) // 2) % 4) // 3,
           80 + 140 * (((i * 23 % 16) // 4) % 4) // 3]
          for i in range(16)]

# Create canvas
canvas = np.zeros((400, 400, 3), dtype=np.uint8)

# Fill each pixel based on perceptron signatures
for y in range(400):
    for x in range(400):
        point = np.array([(x - 200) / 100, (y - 200) / 100])
        signature = sum(p.forward(point) * (2 ** i)
                       for i, p in enumerate(perceptrons))
        canvas[y, x] = colors[signature]

Image.fromarray(canvas).save("perceptron_art.png")
