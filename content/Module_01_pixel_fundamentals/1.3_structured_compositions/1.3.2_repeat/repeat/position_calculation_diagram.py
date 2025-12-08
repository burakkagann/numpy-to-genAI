

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Configuration
N_TILES = 4
TILE_WIDTH = 100
SPACING = 15
CANVAS_HEIGHT = 80

# Create figure
fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
ax.set_xlim(-20, SPACING + N_TILES * TILE_WIDTH + 40)
ax.set_ylim(-90, 220)
ax.set_aspect('equal')
ax.axis('off')

# Title
ax.text(
    (SPACING + N_TILES * TILE_WIDTH) / 2,
    190,
    'Position Calculation Formula',
    ha='center',
    va='top',
    fontsize=16,
    fontweight='bold'
)

# Draw tiles
colors = ['#FFB3BA', '#BAFFC9', '#BAE1FF', '#FFFFBA']  # Pastel colors
for i in range(N_TILES):
    # Calculate position using the formula
    start = SPACING + i * TILE_WIDTH

    # Draw tile
    rect = mpatches.Rectangle(
        (start, CANVAS_HEIGHT),
        TILE_WIDTH,
        CANVAS_HEIGHT,
        linewidth=2,
        edgecolor='black',
        facecolor=colors[i],
        alpha=0.7
    )
    ax.add_patch(rect)

    # Add tile index label
    ax.text(
        start + TILE_WIDTH / 2,
        CANVAS_HEIGHT + 40,
        f'Tile {i}',
        ha='center',
        va='center',
        fontsize=12,
        fontweight='bold'
    )

# Add measurement annotations
# Spacing annotation (before first tile)
ax.plot([0, SPACING], [CANVAS_HEIGHT - 10, CANVAS_HEIGHT - 10], 'k-', lw=1.5)
ax.plot([0, 0], [CANVAS_HEIGHT - 15, CANVAS_HEIGHT - 5], 'k-', lw=1.5)
ax.plot([SPACING, SPACING], [CANVAS_HEIGHT - 15, CANVAS_HEIGHT - 5], 'k-', lw=1.5)
ax.text(
    SPACING / 2,
    CANVAS_HEIGHT - 15,
    f'SPACING\n{SPACING} pixels',
    ha='center',
    va='top',
    fontsize=9,
    color='darkred',
    fontweight='bold'
)

# Tile width annotation (first tile)
start_0 = SPACING
ax.plot([start_0, start_0 + TILE_WIDTH], [CANVAS_HEIGHT + CANVAS_HEIGHT + 10, CANVAS_HEIGHT + CANVAS_HEIGHT + 10], 'k-', lw=1.5)
ax.plot([start_0, start_0], [CANVAS_HEIGHT + CANVAS_HEIGHT + 5, CANVAS_HEIGHT + CANVAS_HEIGHT + 15], 'k-', lw=1.5)
ax.plot([start_0 + TILE_WIDTH, start_0 + TILE_WIDTH], [CANVAS_HEIGHT + CANVAS_HEIGHT + 5, CANVAS_HEIGHT + CANVAS_HEIGHT + 15], 'k-', lw=1.5)
ax.text(
    start_0 + TILE_WIDTH / 2,
    CANVAS_HEIGHT + CANVAS_HEIGHT + 25,
    f'TILE_WIDTH\n{TILE_WIDTH} pixels',
    ha='center',
    va='bottom',
    fontsize=9,
    color='darkblue',
    fontweight='bold'
)

# Formula breakdown for each tile
formula_y = 20
for i in range(N_TILES):
    start = SPACING + i * TILE_WIDTH

    # Position marker
    ax.plot([start, start], [CANVAS_HEIGHT, formula_y + 18], 'k--', lw=1, alpha=0.5)

    # Formula box
    formula_text = f'i = {i}\nstart = {SPACING} + {i} x {TILE_WIDTH}\nstart = {start}'

    box = FancyBboxPatch(
        (start - 40, formula_y - 18),
        80,
        36,
        boxstyle="round,pad=0.05",
        linewidth=1.5,
        edgecolor='black',
        facecolor='lightyellow',
        alpha=0.9
    )
    ax.add_patch(box)

    ax.text(
        start,
        formula_y,
        formula_text,
        ha='center',
        va='center',
        fontsize=6.5,
        family='monospace'
    )

# Add general formula at bottom
general_formula = (
    'General Formula:\n'
    'start = SPACING + (i * TILE_WIDTH)\n'
    'stop = start + TILE_WIDTH\n'
    '\n'
    'Where: i = loop variable (0, 1, 2, 3, ...)'
)
ax.text(
    (SPACING + N_TILES * TILE_WIDTH) / 2,
    -55,
    general_formula,
    ha='center',
    va='bottom',
    fontsize=10,
    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
    family='monospace',
    fontweight='bold'
)

# Add pixel ruler at the bottom for reference
ruler_y = CANVAS_HEIGHT - 35
tick_positions = [0, SPACING, SPACING + TILE_WIDTH, SPACING + 2*TILE_WIDTH,
                  SPACING + 3*TILE_WIDTH, SPACING + 4*TILE_WIDTH]
for pos in tick_positions:
    ax.plot([pos, pos], [ruler_y, ruler_y + 5], 'k-', lw=1)
    ax.text(pos, ruler_y - 5, str(pos), ha='center', va='top', fontsize=7)

ax.plot([0, SPACING + N_TILES * TILE_WIDTH], [ruler_y, ruler_y], 'k-', lw=1.5)
ax.text(-10, ruler_y, 'Pixels:', ha='right', va='center', fontsize=8, style='italic')

plt.tight_layout()
plt.savefig('position_calculation_diagram.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
