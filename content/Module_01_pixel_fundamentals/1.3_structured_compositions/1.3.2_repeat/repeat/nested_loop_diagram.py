
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Configuration
N_TILES = 4
TILE_SIZE = 100
BOTTOM_MARGIN = 120
TOP_MARGIN = 120
SIDE_MARGIN = 60
FIG_WIDTH = N_TILES * TILE_SIZE + 2 * SIDE_MARGIN
FIG_HEIGHT = N_TILES * TILE_SIZE + BOTTOM_MARGIN + TOP_MARGIN

# Create figure
fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
ax.set_xlim(0, FIG_WIDTH)
ax.set_ylim(0, FIG_HEIGHT)
ax.set_aspect('equal')
ax.axis('off')

# Title
ax.text(FIG_WIDTH / 2, FIG_HEIGHT - 20,
        'Nested Loop Execution Order',
        ha='center', va='top', fontsize=16, fontweight='bold')

# Track iteration number
iteration = 0

# Simulate nested loop execution
for y in range(N_TILES):
    for x in range(N_TILES):
        # Calculate tile position (top-left corner)
        tile_x = SIDE_MARGIN + x * TILE_SIZE
        tile_y = FIG_HEIGHT - TOP_MARGIN - (y + 1) * TILE_SIZE  # Flip y-axis for matplotlib

        # Determine color based on execution order (gradient from blue to red)
        color_intensity = iteration / (N_TILES * N_TILES - 1)
        tile_color = plt.cm.coolwarm(color_intensity)

        # Draw tile rectangle
        rect = mpatches.Rectangle(
            (tile_x, tile_y),
            TILE_SIZE,
            TILE_SIZE,
            linewidth=2,
            edgecolor='black',
            facecolor=tile_color,
            alpha=0.7
        )
        ax.add_patch(rect)

        # Add iteration number (large and centered)
        ax.text(
            tile_x + TILE_SIZE / 2,
            tile_y + TILE_SIZE / 2 + 10,
            str(iteration),
            ha='center',
            va='center',
            fontsize=24,
            fontweight='bold',
            color='black'
        )

        # Add (x, y) coordinates below iteration number
        ax.text(
            tile_x + TILE_SIZE / 2,
            tile_y + TILE_SIZE / 2 - 15,
            f'({x},{y})',
            ha='center',
            va='center',
            fontsize=10,
            color='black',
            style='italic'
        )

        iteration += 1

# Add annotations showing loop structure
# Outer loop annotation (left side)
ax.annotate(
    'Outer loop\n(y = 0 to 3)',
    xy=(SIDE_MARGIN - 30, FIG_HEIGHT - TOP_MARGIN - TILE_SIZE * 2),
    fontsize=11,
    ha='right',
    va='center',
    color='darkred',
    fontweight='bold'
)

# Add arrows showing outer loop progression
for y in range(N_TILES - 1):
    arrow_y = FIG_HEIGHT - TOP_MARGIN - (y + 0.5) * TILE_SIZE
    ax.annotate(
        '',
        xy=(SIDE_MARGIN - 35, FIG_HEIGHT - TOP_MARGIN - (y + 1.5) * TILE_SIZE),
        xytext=(SIDE_MARGIN - 35, arrow_y),
        arrowprops=dict(arrowstyle='->', color='darkred', lw=2)
    )

# Inner loop annotation (top)
ax.annotate(
    'Inner loop (x = 0 to 3)',
    xy=(SIDE_MARGIN + TILE_SIZE * 2, FIG_HEIGHT - TOP_MARGIN + 60),
    fontsize=11,
    ha='center',
    va='bottom',
    color='darkblue',
    fontweight='bold'
)

# Add arrows showing inner loop progression for first row
arrow_y = FIG_HEIGHT - TOP_MARGIN + 55
for x in range(N_TILES - 1):
    arrow_x = SIDE_MARGIN + (x + 0.5) * TILE_SIZE
    ax.annotate(
        '',
        xy=(SIDE_MARGIN + (x + 1.5) * TILE_SIZE, arrow_y),
        xytext=(arrow_x, arrow_y),
        arrowprops=dict(arrowstyle='->', color='darkblue', lw=2)
    )

# Add execution flow description
execution_text = (
    "Execution flow:\n"
    "1. Outer loop sets y=0\n"
    "2. Inner loop runs x=0,1,2,3 (iterations 0-3)\n"
    "3. Outer loop advances to y=1\n"
    "4. Inner loop runs x=0,1,2,3 (iterations 4-7)\n"
    "... and so on"
)
ax.text(
    FIG_WIDTH / 2,
    10,
    execution_text,
    ha='center',
    va='bottom',
    fontsize=9,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
    family='monospace'
)

plt.tight_layout()
plt.savefig('nested_loop_execution.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
