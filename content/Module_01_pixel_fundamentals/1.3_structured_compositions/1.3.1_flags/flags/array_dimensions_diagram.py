
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D

# Create figure with 3D projection
fig = plt.figure(figsize=(12, 8), dpi=150)
ax = fig.add_subplot(111, projection='3d')

# Define small array dimensions for visualization
height, width = 4, 6
channel_spacing = 1.5

# Create three layers representing R, G, B channels
channels = ['Red', 'Green', 'Blue']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
z_positions = [0, channel_spacing, channel_spacing * 2]

for idx, (channel_name, color, z_pos) in enumerate(zip(channels, colors, z_positions)):
    # Create a grid for this channel
    for row in range(height):
        for col in range(width):
            # Create cube/rectangle for each cell
            x = [col, col + 0.9, col + 0.9, col, col]
            y = [row, row, row + 0.9, row + 0.9, row]
            z = [z_pos] * 5

            # Draw cell
            ax.plot(x, y, z, color=color, alpha=0.6, linewidth=1.5)

            # Fill cell
            vertices = [(col, row, z_pos),
                       (col + 0.9, row, z_pos),
                       (col + 0.9, row + 0.9, z_pos),
                       (col, row + 0.9, z_pos)]
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            poly = Poly3DCollection([vertices], alpha=0.3, facecolor=color, edgecolor=color)
            ax.add_collection3d(poly)

    # Add channel label
    ax.text(width / 2, -0.8, z_pos, channel_name,
            fontsize=14, fontweight='bold', color=color,
            ha='center', va='top')

# Add dimension annotations
ax.text(width / 2, -2, -0.5, 'Width (columns)',
        fontsize=12, fontweight='bold', color='#333333',
        ha='center', va='top')

ax.text(-1.5, height / 2, -0.5, 'Height\n(rows)',
        fontsize=12, fontweight='bold', color='#333333',
        ha='center', va='center')

ax.text(width + 1, height + 1, channel_spacing,
        'Channels\n(RGB)',
        fontsize=12, fontweight='bold', color='#333333',
        ha='left', va='center')

# Add indexing example
ax.text(width / 2, height + 2, channel_spacing * 2 + 0.5,
        'Shape: (height, width, 3)\nIndexing: image[row, col, channel]',
        fontsize=13, fontweight='bold', color='#0066CC',
        ha='center', va='bottom',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='white', edgecolor='#0066CC', linewidth=2))

# Add example pixel
example_row, example_col = 1, 2
for idx, z_pos in enumerate(z_positions):
    ax.scatter([example_col + 0.45], [example_row + 0.45], [z_pos],
               color='yellow', s=200, marker='o', edgecolors='black', linewidths=2, zorder=10)

ax.text(example_col + 0.45, example_row + 0.45, channel_spacing * 2 + 0.8,
        f'Pixel at [{example_row}, {example_col}, :]',
        fontsize=10, fontweight='bold', color='#FF4500',
        ha='center', va='bottom')

# Set viewing angle
ax.view_init(elev=20, azim=45)

# Set axis labels
ax.set_xlabel('Column (x)', fontsize=11, fontweight='bold')
ax.set_ylabel('Row (y)', fontsize=11, fontweight='bold')
ax.set_zlabel('Channel', fontsize=11, fontweight='bold')

# Set axis limits
ax.set_xlim(-1, width + 1)
ax.set_ylim(-2, height + 1)
ax.set_zlim(-0.5, channel_spacing * 2 + 1)

# Title
ax.set_title('RGB Image as 3D NumPy Array', fontsize=16, fontweight='bold', pad=20)

# Remove grid for cleaner look
ax.grid(False)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Save figure
plt.tight_layout()
plt.savefig('array_dimensions_diagram.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("Array dimensions diagram created successfully!")
print("Output saved as: array_dimensions_diagram.png")
