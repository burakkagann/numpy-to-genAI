import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# =============================================================================
# Create figure with two panels
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

# =============================================================================
# Panel 1: Conceptual diagram showing SDF values
# =============================================================================
ax1.set_xlim(-3, 3)
ax1.set_ylim(-3, 3)
ax1.set_aspect('equal')
ax1.set_title('Signed Distance Function Concept', fontsize=14, fontweight='bold')

# Draw the shape boundary (circle)
circle = plt.Circle((0, 0), 1.5, fill=False, color='black', linewidth=3)
ax1.add_patch(circle)

# Add shaded regions
# Inside region (light blue)
inside_circle = plt.Circle((0, 0), 1.5, fill=True, color='lightblue', alpha=0.5)
ax1.add_patch(inside_circle)

# Sample points with their SDF values
sample_points = [
    (0, 0, -1.5, 'Inside\nd = -1.5'),       # Center
    (0.8, 0.8, -0.37, 'Inside\nd = -0.37'),  # Inside
    (0, 1.5, 0, 'Boundary\nd = 0'),          # On boundary
    (2.0, 0, 0.5, 'Outside\nd = +0.5'),      # Outside
    (2.2, 2.2, 1.61, 'Outside\nd = +1.61'),  # Far outside
]

for x, y, d, label in sample_points:
    # Draw point
    color = 'blue' if d < 0 else ('green' if d == 0 else 'red')
    ax1.plot(x, y, 'o', markersize=12, color=color, markeredgecolor='black')

    # Draw distance line to nearest boundary point
    if d != 0:
        # Calculate nearest point on circle
        dist_to_center = np.sqrt(x**2 + y**2)
        if dist_to_center > 0:
            nearest_x = x * 1.5 / dist_to_center
            nearest_y = y * 1.5 / dist_to_center
            ax1.plot([x, nearest_x], [y, nearest_y], '--', color=color, alpha=0.7, linewidth=1.5)

    # Add label with offset
    offset_x = 0.3 if x >= 0 else -0.8
    offset_y = 0.2
    ax1.annotate(label, (x, y), (x + offset_x, y + offset_y), fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Add legend
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Inside (d < 0)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Boundary (d = 0)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Outside (d > 0)')
]
ax1.legend(handles=legend_elements, loc='lower right', fontsize=10)

ax1.set_xlabel('X', fontsize=11)
ax1.set_ylabel('Y', fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='gray', linewidth=0.5)
ax1.axvline(x=0, color='gray', linewidth=0.5)

# =============================================================================
# Panel 2: SDF cross-section showing values along a line
# =============================================================================
# Create 1D cross-section through center
x_vals = np.linspace(-3, 3, 200)
circle_radius = 1.5

# SDF for circle: distance - radius
sdf_values = np.abs(x_vals) - circle_radius

ax2.set_title('SDF Cross-Section (y = 0)', fontsize=14, fontweight='bold')
ax2.fill_between(x_vals, sdf_values, 0, where=(sdf_values < 0),
                 color='lightblue', alpha=0.7, label='Inside (d < 0)')
ax2.fill_between(x_vals, sdf_values, 0, where=(sdf_values > 0),
                 color='lightsalmon', alpha=0.7, label='Outside (d > 0)')
ax2.plot(x_vals, sdf_values, 'k-', linewidth=2, label='SDF value')
ax2.axhline(y=0, color='green', linewidth=2, linestyle='--', label='Boundary (d = 0)')

# Mark boundary points
ax2.plot([-1.5, 1.5], [0, 0], 'go', markersize=10, markeredgecolor='black')

# Annotations
ax2.annotate('Shape edge', (-1.5, 0), (-2.2, 0.8), fontsize=10,
            arrowprops=dict(arrowstyle='->', color='green'))
ax2.annotate('Shape edge', (1.5, 0), (2.2, 0.8), fontsize=10,
            arrowprops=dict(arrowstyle='->', color='green'))
ax2.annotate('Inside shape', (0, -0.75), fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
ax2.annotate('Outside', (-2.5, 0.5), fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightsalmon', alpha=0.8))
ax2.annotate('Outside', (2.5, 0.5), fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightsalmon', alpha=0.8))

ax2.set_xlabel('X Position', fontsize=11)
ax2.set_ylabel('Signed Distance Value', fontsize=11)
ax2.set_ylim(-2, 2)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right', fontsize=9)

# =============================================================================
# Save figure
# =============================================================================
plt.tight_layout()
plt.savefig('sdf_concept_diagram.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

