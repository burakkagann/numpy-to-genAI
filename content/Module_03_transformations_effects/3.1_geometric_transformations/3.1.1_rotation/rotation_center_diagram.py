import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# Create a simple arrow shape to make rotation direction clear
def create_arrow_shape(size=100):
    """Create an arrow pointing right on a transparent background."""
    shape = np.zeros((size, size, 4), dtype=np.uint8)

    # Arrow body (horizontal rectangle)
    body_top, body_bottom = 35, 65
    body_left, body_right = 10, 60
    shape[body_top:body_bottom, body_left:body_right] = [0, 150, 200, 255]  # Cyan

    # Arrow head (triangle on the right)
    for y in range(size):
        for x in range(60, 90):
            # Create triangular head pointing right
            dist_from_center = abs(y - 50)
            if dist_from_center < (90 - x):
                shape[y, x] = [0, 150, 200, 255]

    return shape

# Create figure with two comparison panels
fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='#1a1a2e')

# Panel 1: Rotation around image center (default behavior)
ax1 = axes[0]
ax1.set_facecolor('#1a1a2e')
ax1.set_title('Rotation Around Image Center', color='white', fontsize=14, fontweight='bold', pad=15)

# Draw original and rotated shapes
arrow = create_arrow_shape()
rotated_45 = ndimage.rotate(arrow, 45, reshape=True, mode='constant', cval=0)
rotated_90 = ndimage.rotate(arrow, 90, reshape=True, mode='constant', cval=0)

# Plot center point and shapes
ax1.plot(50, 50, 'ro', markersize=12, label='Center of Rotation', zorder=10)

# Show ghost arrows at different angles to indicate rotation path
for angle in [0, 30, 60, 90]:
    rotated = ndimage.rotate(arrow[:,:,:3], angle, reshape=False, mode='constant', cval=0)
    alpha = 0.3 if angle != 0 else 1.0
    ax1.imshow(rotated, extent=[0, 100, 0, 100], alpha=alpha, origin='lower')

# Draw rotation arc
theta = np.linspace(0, np.pi/2, 50)
arc_r = 30
arc_x = 50 + arc_r * np.cos(theta)
arc_y = 50 + arc_r * np.sin(theta)
ax1.plot(arc_x, arc_y, 'w--', linewidth=2, alpha=0.7)
ax1.annotate('', xy=(arc_x[-1], arc_y[-1]), xytext=(arc_x[-5], arc_y[-5]),
            arrowprops=dict(arrowstyle='->', color='white', lw=2))

ax1.set_xlim(-10, 110)
ax1.set_ylim(-10, 110)
ax1.set_aspect('equal')
ax1.axis('off')
ax1.text(50, -5, 'Shape stays centered', color='#aaa', ha='center', fontsize=11)

# Panel 2: Rotation around corner (offset center)
ax2 = axes[1]
ax2.set_facecolor('#1a1a2e')
ax2.set_title('Rotation Around Corner Point', color='white', fontsize=14, fontweight='bold', pad=15)

# For corner rotation, we'll simulate by translating, rotating, then translating back
# This shows the shape orbiting around the corner
corner_x, corner_y = 10, 50

# Plot corner point
ax2.plot(corner_x, corner_y, 'yo', markersize=12, label='Center of Rotation', zorder=10)

# Show arrows at different angles orbiting around corner
for angle in [0, 30, 60, 90]:
    # Calculate position on circular path
    radius = 40  # Distance from corner to shape center
    rad_angle = np.radians(angle)

    # Position of shape center after rotation around corner
    shape_center_x = corner_x + radius * np.cos(rad_angle)
    shape_center_y = corner_y + radius * np.sin(rad_angle)

    # Rotate the arrow itself
    rotated = ndimage.rotate(arrow[:,:,:3], angle, reshape=False, mode='constant', cval=0)

    # Calculate extent for positioning
    half_size = 50
    extent = [shape_center_x - half_size, shape_center_x + half_size,
              shape_center_y - half_size, shape_center_y + half_size]

    alpha = 0.3 if angle != 0 else 1.0
    ax2.imshow(rotated, extent=extent, alpha=alpha, origin='lower')

# Draw rotation arc from corner
arc_r = 40
arc_x = corner_x + arc_r * np.cos(theta)
arc_y = corner_y + arc_r * np.sin(theta)
ax2.plot(arc_x, arc_y, 'w--', linewidth=2, alpha=0.7)
ax2.annotate('', xy=(arc_x[-1], arc_y[-1]), xytext=(arc_x[-5], arc_y[-5]),
            arrowprops=dict(arrowstyle='->', color='white', lw=2))

ax2.set_xlim(-50, 150)
ax2.set_ylim(-10, 110)
ax2.set_aspect('equal')
ax2.axis('off')
ax2.text(50, -5, 'Shape orbits around pivot', color='#aaa', ha='center', fontsize=11)

# Add legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Image Center', linestyle='None'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='Corner Pivot', linestyle='None'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=2,
           frameon=False, fontsize=11, labelcolor='white')

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig('rotation_center_diagram.png', dpi=150, facecolor='#1a1a2e',
            edgecolor='none', bbox_inches='tight')
plt.close()