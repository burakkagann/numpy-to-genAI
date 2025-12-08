
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# =============================================================================
# Configuration
# =============================================================================
GRID_SIZE = 10          # Number of cells in each direction
CENTER = (5, 5)         # Center of the circle in grid coordinates
RADIUS = 3.5            # Circle radius in grid units
FIG_SIZE = (10, 8)      # Figure size in inches
DPI = 150               # Output resolution

# =============================================================================
# Create figure and axis
# =============================================================================
fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)

# =============================================================================
# Draw the grid and classify pixels as inside/outside circle
# =============================================================================
for row in range(GRID_SIZE):
    for col in range(GRID_SIZE):
        # Calculate distance from this cell's center to the circle center
        cell_center_x = col + 0.5
        cell_center_y = row + 0.5
        distance = np.sqrt((cell_center_x - CENTER[0]) ** 2 +
                          (cell_center_y - CENTER[1]) ** 2)

        # Determine color based on whether pixel is inside circle
        if distance < RADIUS:
            color = '#FFD700'       # Gold for inside
            edge_color = '#FF4500'  # Orange-red border
        else:
            color = '#E8E8E8'       # Light gray for outside
            edge_color = '#CCCCCC'  # Gray border

        # Draw the cell
        rect = mpatches.Rectangle(
            (col, row), 1, 1,
            linewidth=1,
            edgecolor=edge_color,
            facecolor=color
        )
        ax.add_patch(rect)

# =============================================================================
# Draw the mathematical circle (perfect, not pixelated)
# =============================================================================
circle = plt.Circle(
    CENTER, RADIUS,
    fill=False,
    color='#2E86AB',
    linewidth=3,
    linestyle='--',
    label='Mathematical circle'
)
ax.add_patch(circle)

# =============================================================================
# Mark the center point
# =============================================================================
ax.plot(CENTER[0], CENTER[1], 'ko', markersize=12, zorder=5)
ax.annotate(
    f'Center\n(cx, cy) = {CENTER}',
    xy=CENTER,
    xytext=(CENTER[0] + 2.5, CENTER[1] + 2.5),
    fontsize=11,
    ha='left',
    arrowprops=dict(arrowstyle='->', color='black', lw=1.5)
)

# =============================================================================
# Illustrate distance calculation for a sample point
# =============================================================================
sample_point = (7.5, 6.5)  # A point near the circle edge

# Draw the sample point
ax.plot(sample_point[0], sample_point[1], 'r*', markersize=15, zorder=5)

# Draw distance components (dx, dy, d)
# Horizontal line (dx)
ax.plot([CENTER[0], sample_point[0]], [CENTER[1], CENTER[1]],
        'g-', linewidth=2, label='dx = x - cx')
# Vertical line (dy)
ax.plot([sample_point[0], sample_point[0]], [CENTER[1], sample_point[1]],
        'm-', linewidth=2, label='dy = y - cy')
# Direct distance (d)
ax.plot([CENTER[0], sample_point[0]], [CENTER[1], sample_point[1]],
        'r-', linewidth=2, alpha=0.7, label='d = distance')

# Add distance formula annotation
distance_to_sample = np.sqrt((sample_point[0] - CENTER[0]) ** 2 +
                             (sample_point[1] - CENTER[1]) ** 2)
ax.annotate(
    f'Sample pixel (x, y)\n'
    f'd = sqrt(dx² + dy²)\n'
    f'd = {distance_to_sample:.2f}',
    xy=sample_point,
    xytext=(sample_point[0] + 0.3, sample_point[1] + 1.5),
    fontsize=10,
    ha='left',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
    arrowprops=dict(arrowstyle='->', color='red', lw=1.5)
)

# =============================================================================
# Add labels for dx and dy
# =============================================================================
mid_x = (CENTER[0] + sample_point[0]) / 2
ax.text(mid_x, CENTER[1] - 0.3, 'dx', fontsize=12, color='green',
        fontweight='bold', ha='center')

mid_y = (CENTER[1] + sample_point[1]) / 2
ax.text(sample_point[0] + 0.2, mid_y, 'dy', fontsize=12, color='purple',
        fontweight='bold', va='center')

# =============================================================================
# Draw the radius annotation
# =============================================================================
radius_end = (CENTER[0] + RADIUS * np.cos(np.pi/4),
              CENTER[1] + RADIUS * np.sin(np.pi/4))
ax.annotate(
    f'r = {RADIUS}',
    xy=((CENTER[0] + radius_end[0])/2, (CENTER[1] + radius_end[1])/2),
    fontsize=11,
    color='#2E86AB',
    fontweight='bold'
)

# =============================================================================
# Configure axes and labels
# =============================================================================
ax.set_xlim(-0.5, GRID_SIZE + 0.5)
ax.set_ylim(-0.5, GRID_SIZE + 0.5)
ax.set_aspect('equal')

# Add axis labels with tick marks
ax.set_xticks(range(GRID_SIZE + 1))
ax.set_yticks(range(GRID_SIZE + 1))
ax.set_xlabel('X (columns)', fontsize=12)
ax.set_ylabel('Y (rows)', fontsize=12)

# Add title
ax.set_title(
    'Distance Formula: Determining Pixels Inside a Circle\n'
    'Pixel is INSIDE if: d < r  (gold cells)',
    fontsize=14,
    fontweight='bold'
)

# Add legend
legend_elements = [
    mpatches.Patch(facecolor='#FFD700', edgecolor='#FF4500',
                   label='Inside circle (d < r)'),
    mpatches.Patch(facecolor='#E8E8E8', edgecolor='#CCCCCC',
                   label='Outside circle (d >= r)'),
    plt.Line2D([0], [0], color='#2E86AB', linewidth=2, linestyle='--',
               label='Mathematical circle boundary')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

# Add formula box
formula_text = (
    "Distance Formula:\n"
    "d = sqrt((x - cx)² + (y - cy)²)\n\n"
    "Optimization (avoid sqrt):\n"
    "d² < r²  is equivalent to  d < r"
)
ax.text(
    0.02, 0.98, formula_text,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
    family='monospace'
)

# =============================================================================
# Save the diagram
# =============================================================================
plt.tight_layout()
plt.savefig('distance_formula_diagram.png', dpi=DPI, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
