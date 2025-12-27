"""
Exercise 2, Goal 3: Layered activations in a single channel
Shows how combining multiple sigmoid activations creates concentric rings.
"""
import numpy as np
from PIL import Image

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

width, height = 512, 512

x = np.linspace(-5, 5, width)
y = np.linspace(-5, 5, height)
X, Y = np.meshgrid(x, y)

distance = np.sqrt(X**2 + Y**2)

# Layered sigmoids at different radii
layer1 = sigmoid(distance - 1)
layer2 = sigmoid(distance - 3)
layer3 = sigmoid(distance - 5)

# Combine all layers into a single grayscale channel
combined = (layer1 + layer2 + layer3) / 3 * 255

# Create a monochromatic (purple/blue) image for visual appeal
image_array = np.zeros((height, width, 3), dtype=np.uint8)
image_array[:, :, 0] = (combined * 0.6).astype(np.uint8)  # Red (lower)
image_array[:, :, 1] = (combined * 0.2).astype(np.uint8)  # Green (lower)
image_array[:, :, 2] = combined.astype(np.uint8)          # Blue (full)

output = Image.fromarray(image_array, mode='RGB')
output.save('exercise2_goal3_layered.png')
print("Saved exercise2_goal3_layered.png")
