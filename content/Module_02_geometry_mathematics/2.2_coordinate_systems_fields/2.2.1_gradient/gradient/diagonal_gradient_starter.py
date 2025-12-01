"""
Exercise 3 Starter Code: Create a Diagonal Gradient

Your task: Create a 400x400 grayscale diagonal gradient that transitions
from black (top-left corner) to white (bottom-right corner).

Hint: A diagonal gradient combines horizontal and vertical gradients.

Author: Claude (NumPy-to-GenAI Project)
Date: 2025-01-30
"""

import numpy as np
from PIL import Image

# Step 1: Define image dimensions
size = 400

# Step 2: Create horizontal gradient values (0 to 255 across width)
# TODO: Use np.linspace() to create values from 0 to 255 with 'size' points
horizontal = None  # Replace with your code

# Step 3: Create vertical gradient values (0 to 255 down height)
# TODO: Use np.linspace() with reshape(-1, 1) to make it a column vector
vertical = None  # Replace with your code

# Step 4: Combine horizontal and vertical components
# TODO: Average the two components to create diagonal effect
# Hint: (horizontal + vertical) / 2
diagonal = None  # Replace with your code

# Step 5: Convert to uint8 and save
# TODO: Convert diagonal to uint8 dtype
gradient_image = None  # Replace with your code

# Save the result
output = Image.fromarray(gradient_image, mode='L')
output.save('my_diagonal_gradient.png')
print("Saved as my_diagonal_gradient.png")
