"""
Exercise 2, Goal 1: Zoomed coordinate range
Shows how changing the linspace range affects the visualization.
"""
import numpy as np
from PIL import Image

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh_activation(x):
    return np.tanh(x)

width, height = 512, 512

# Zoomed in range: -2 to 2 (shows more detail of the center)
x = np.linspace(-2, 2, width)
y = np.linspace(-2, 2, height)
X, Y = np.meshgrid(x, y)

distance = np.sqrt(X**2 + Y**2)

red_channel = sigmoid(distance - 2.5) * 255
green_channel = relu(3 - distance) / 3 * 255
blue_channel = (tanh_activation(distance - 2) + 1) / 2 * 255

image_array = np.zeros((height, width, 3), dtype=np.uint8)
image_array[:, :, 0] = red_channel.astype(np.uint8)
image_array[:, :, 1] = green_channel.astype(np.uint8)
image_array[:, :, 2] = blue_channel.astype(np.uint8)

output = Image.fromarray(image_array, mode='RGB')
output.save('exercise2_goal1_zoomed.png')
print("Saved exercise2_goal1_zoomed.png")
