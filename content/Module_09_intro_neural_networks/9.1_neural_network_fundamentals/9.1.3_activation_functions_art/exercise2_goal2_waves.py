"""
Exercise 2, Goal 2: Wave pattern input
Shows how using sin/cos creates interesting wave patterns.
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

x = np.linspace(-5, 5, width)
y = np.linspace(-5, 5, height)
X, Y = np.meshgrid(x, y)

# Wave pattern instead of radial distance
wave = np.sin(X * 2) + np.cos(Y * 2)

red_channel = sigmoid(wave) * 255
green_channel = relu(wave + 1) * 64
blue_channel = (tanh_activation(wave) + 1) / 2 * 255

image_array = np.zeros((height, width, 3), dtype=np.uint8)
image_array[:, :, 0] = np.clip(red_channel, 0, 255).astype(np.uint8)
image_array[:, :, 1] = np.clip(green_channel, 0, 255).astype(np.uint8)
image_array[:, :, 2] = np.clip(blue_channel, 0, 255).astype(np.uint8)

output = Image.fromarray(image_array, mode='RGB')
output.save('exercise2_goal2_waves.png')
print("Saved exercise2_goal2_waves.png")
