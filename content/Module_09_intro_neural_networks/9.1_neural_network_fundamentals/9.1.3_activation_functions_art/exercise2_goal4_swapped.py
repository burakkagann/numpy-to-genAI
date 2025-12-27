"""
Exercise 2, Goal 4: Swapped channel assignments
Shows how using all sigmoid activations at different offsets creates a sunset effect.
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

# All same activation (sigmoid) but different offsets
# Creates a monochromatic/gradient effect
red_channel = sigmoid(distance - 1) * 255
green_channel = sigmoid(distance - 2.5) * 255
blue_channel = sigmoid(distance - 4) * 255

image_array = np.zeros((height, width, 3), dtype=np.uint8)
image_array[:, :, 0] = red_channel.astype(np.uint8)
image_array[:, :, 1] = green_channel.astype(np.uint8)
image_array[:, :, 2] = blue_channel.astype(np.uint8)

output = Image.fromarray(image_array, mode='RGB')
output.save('exercise2_goal4_swapped.png')
print("Saved exercise2_goal4_swapped.png")
