

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Create figure
fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

# Create 5x5 grid
grid_size = 5

# Draw grid cells
for row in range(grid_size):
    for col in range(grid_size):
        # Determine if this cell is in the sliced region [1:4, 2:5]
        if 1 <= row < 4 and 2 <= col < 5:
            color = '#FFD700'  # Gold for selected region
            edgecolor = '#FF4500'  # Orange-red border
            linewidth = 3
        else:
            color = '#E8E8E8'  # Light gray for unselected
            edgecolor = '#888888'  # Dark gray border
            linewidth = 1

        # Draw rectangle
        rect = mpatches.Rectangle((col, grid_size - row - 1), 1, 1,
                                   linewidth=linewidth,
                                   edgecolor=edgecolor,
                                   facecolor=color)
        ax.add_patch(rect)

        # Add cell coordinates as text
        ax.text(col + 0.5, grid_size - row - 0.5, f'[{row},{col}]',
                ha='center', va='center', fontsize=10, fontweight='bold')

# Add row labels (0-4)
for row in range(grid_size):
    ax.text(-0.3, grid_size - row - 0.5, f'{row}',
            ha='center', va='center', fontsize=12, fontweight='bold', color='#0066CC')

# Add column labels (0-4)
for col in range(grid_size):
    ax.text(col + 0.5, grid_size + 0.3, f'{col}',
            ha='center', va='center', fontsize=12, fontweight='bold', color='#0066CC')

# Add dimension labels
ax.text(-1.2, grid_size / 2, 'Rows\n(dimension 0)',
        ha='center', va='center', fontsize=14, fontweight='bold',
        rotation=90, color='#0066CC')
ax.text(grid_size / 2, grid_size + 1, 'Columns (dimension 1)',
        ha='center', va='center', fontsize=14, fontweight='bold', color='#0066CC')

# Add slicing notation annotations
ax.annotate('image[1:4, 2:5]', xy=(3.5, 2.5), xytext=(6.5, 3.5),
            fontsize=16, fontweight='bold', color='#FF4500',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#FF4500', linewidth=2),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3',
                           color='#FF4500', lw=2))

# Add explanation text
ax.text(grid_size / 2, -1.2,
        'Slice notation: [start:stop] (start is inclusive, stop is exclusive)\n' +
        'image[1:4, 2:5] selects rows 1,2,3 and columns 2,3,4',
        ha='center', va='top', fontsize=11,
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#F0F0F0', edgecolor='#888888'))

# Set axis properties
ax.set_xlim(-1.5, grid_size + 2)
ax.set_ylim(-1.8, grid_size + 1.5)
ax.set_aspect('equal')
ax.axis('off')

# Add title
ax.set_title('NumPy Array Slicing Syntax', fontsize=18, fontweight='bold', pad=20)

# Save figure
plt.tight_layout()
plt.savefig('slicing_syntax_diagram.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
