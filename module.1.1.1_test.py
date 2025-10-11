import numpy as np
import matplotlib.pyplot as plt

# Create image
height, width = 200, 200
image = np.zeros((height, width, 3), dtype=np.uint8)

# Create gradient from red (left) to blue (right)
for col in range(width):
    image[:, col, 0] = 255 - (col * 255 // width)  # Red decreases
    image[:, col, 2] = col * 255 // width          # Blue increases
    # Green channel stays 0

# Display
plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.axis('off')
plt.title('Red to Blue Gradient')
plt.show()