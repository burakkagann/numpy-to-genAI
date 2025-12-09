import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Create smaller image for clearer arrow visualization
height, width = 256, 256
center_x, center_y = width // 2, height // 2

# Create coordinate arrays
y_coords, x_coords = np.ogrid[:height, :width]
dx = center_x - x_coords  # Vector pointing toward center
dy = center_y - y_coords

# Calculate angle and create color-coded image
angle = np.arctan2(dy, dx)
hue = ((angle + np.pi) / (2 * np.pi) * 255).astype(np.uint8)

# Create RGB image using HSV-style color wheel
image = np.zeros((height, width, 3), dtype=np.uint8)
segment = hue // 43
remainder = (hue % 43) * 6

image[:, :, 0] = np.where(segment == 0, 255,
                 np.where(segment == 1, 255 - remainder,
                 np.where(segment == 2, 0,
                 np.where(segment == 3, 0,
                 np.where(segment == 4, remainder, 255)))))

image[:, :, 1] = np.where(segment == 0, remainder,
                 np.where(segment == 1, 255,
                 np.where(segment == 2, 255,
                 np.where(segment == 3, 255 - remainder,
                 np.where(segment == 4, 0, 0)))))

image[:, :, 2] = np.where(segment == 0, 0,
                 np.where(segment == 1, 0,
                 np.where(segment == 2, remainder,
                 np.where(segment == 3, 255,
                 np.where(segment == 4, 255, 255 - remainder)))))

# Create figure with two panels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left panel: Color-coded field only
ax1.imshow(image)
ax1.set_title('Color-Coded Vector Field', fontsize=14, fontweight='bold')
ax1.set_xlabel('X coordinate')
ax1.set_ylabel('Y coordinate')
ax1.plot(center_x, center_y, 'ko', markersize=8, label='Center')
ax1.legend(loc='upper right')

# Right panel: Color field with arrow overlay
ax2.imshow(image, alpha=0.7)

# Create arrow grid (sample every 20 pixels for clarity)
step = 20
arrow_x = np.arange(step//2, width, step)
arrow_y = np.arange(step//2, height, step)
X, Y = np.meshgrid(arrow_x, arrow_y)

# Calculate direction vectors at arrow positions
U = center_x - X  # dx component
V = center_y - Y  # dy component

# Normalize arrows for consistent length
magnitude = np.sqrt(U**2 + V**2)
magnitude[magnitude == 0] = 1  # Avoid division by zero
U_norm = U / magnitude * 12  # Scale for visibility
V_norm = V / magnitude * 12

# Plot arrows (note: matplotlib Y-axis is inverted for images)
ax2.quiver(X, Y, U_norm, -V_norm, color='white',
           width=0.003, headwidth=4, headlength=5,
           edgecolor='black', linewidth=0.5)

ax2.set_title('Vector Field with Direction Arrows', fontsize=14, fontweight='bold')
ax2.set_xlabel('X coordinate')
ax2.set_ylabel('Y coordinate')
ax2.plot(center_x, center_y, 'ko', markersize=8)

# Add annotation explaining the concept
fig.suptitle('Understanding Vector Fields: Each Pixel Has a Direction',
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('vector_field_diagram.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

