"""
Delaunay Triangulation Concept Diagram

Creates a 2x2 visualization explaining key concepts:
1. Basic Delaunay triangulation
2. Circumcircle property (no point inside any circumcircle)
3. Grid-based vs random point distributions
4. Why Delaunay avoids thin triangles

Author: Claude (NumPy-to-GenAI Project)
Date: 2025-12-02
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial import Delaunay

def circumcircle(p1, p2, p3):
    """Calculate circumcircle center and radius for a triangle."""
    ax, ay = p1
    bx, by = p2
    cx, cy = p3

    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-10:
        return None, None

    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d

    radius = np.sqrt((ax - ux)**2 + (ay - uy)**2)
    return (ux, uy), radius

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
fig.suptitle('Understanding Delaunay Triangulation', fontsize=16, fontweight='bold', y=0.98)

# Panel 1: Basic Delaunay triangulation
ax1 = axes[0, 0]
np.random.seed(42)
points1 = np.random.rand(30, 2) * 100
tri1 = Delaunay(points1)
ax1.triplot(points1[:, 0], points1[:, 1], tri1.simplices, color='steelblue', linewidth=1, linestyle='-')
ax1.plot(points1[:, 0], points1[:, 1], 'o', color='coral', markersize=8)
ax1.set_title('(a) Delaunay Triangulation\n30 random points', fontsize=12)
ax1.set_xlim(-5, 105)
ax1.set_ylim(-5, 105)
ax1.set_aspect('equal')
ax1.axis('off')

# Panel 2: Circumcircle property demonstration
ax2 = axes[0, 1]
np.random.seed(123)
points2 = np.random.rand(12, 2) * 100
tri2 = Delaunay(points2)
ax2.triplot(points2[:, 0], points2[:, 1], tri2.simplices, color='steelblue', linewidth=1, linestyle='-')
ax2.plot(points2[:, 0], points2[:, 1], 'o', color='coral', markersize=8)

# Draw circumcircles for a few triangles
for i, simplex in enumerate(tri2.simplices[:4]):  # Show 4 circumcircles
    p1, p2, p3 = points2[simplex[0]], points2[simplex[1]], points2[simplex[2]]
    center, radius = circumcircle(p1, p2, p3)
    if center is not None:
        circle = Circle(center, radius, fill=False, color='green',
                       linestyle='--', linewidth=1.5, alpha=0.7)
        ax2.add_patch(circle)
        ax2.plot(*center, '+', color='green', markersize=8, markeredgewidth=2)

ax2.set_title('(b) Circumcircle Property\nNo point inside any circumcircle', fontsize=12)
ax2.set_xlim(-20, 120)
ax2.set_ylim(-20, 120)
ax2.set_aspect('equal')
ax2.axis('off')

# Panel 3: Grid-based points
ax3 = axes[1, 0]
# Create grid points with slight randomness
grid_x, grid_y = np.meshgrid(np.linspace(5, 95, 6), np.linspace(5, 95, 6))
points3 = np.column_stack([grid_x.ravel(), grid_y.ravel()])
# Add slight jitter
np.random.seed(42)
points3 += np.random.randn(*points3.shape) * 3
tri3 = Delaunay(points3)
ax3.triplot(points3[:, 0], points3[:, 1], tri3.simplices, color='steelblue', linewidth=1, linestyle='-')
ax3.plot(points3[:, 0], points3[:, 1], 'o', color='coral', markersize=8)
ax3.set_title('(c) Grid-Based Distribution\nRegular patterns with jitter', fontsize=12)
ax3.set_xlim(-5, 105)
ax3.set_ylim(-5, 105)
ax3.set_aspect('equal')
ax3.axis('off')

# Panel 4: Comparison - good vs thin triangles
ax4 = axes[1, 1]
# Create points that would make thin triangles without Delaunay
np.random.seed(99)
points4 = np.random.rand(25, 2) * 100
tri4 = Delaunay(points4)

# Calculate minimum angles for visualization
def min_angle(p1, p2, p3):
    """Calculate minimum angle in a triangle."""
    def angle(a, b, c):
        ba = a - b
        bc = c - b
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-10)
        return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
    return min(angle(p1, p2, p3), angle(p2, p3, p1), angle(p3, p1, p2))

# Color triangles by minimum angle
from matplotlib.collections import PolyCollection
triangles = []
colors = []
for simplex in tri4.simplices:
    triangle = points4[simplex]
    triangles.append(triangle)
    min_ang = min_angle(triangle[0], triangle[1], triangle[2])
    colors.append(min_ang)

colors = np.array(colors)
collection = PolyCollection(triangles, array=colors, cmap='RdYlGn',
                           edgecolors='steelblue', linewidths=1)
ax4.add_collection(collection)
ax4.plot(points4[:, 0], points4[:, 1], 'o', color='coral', markersize=8)
ax4.set_title('(d) Triangle Quality\nColored by minimum angle (green=good)', fontsize=12)
ax4.set_xlim(-5, 105)
ax4.set_ylim(-5, 105)
ax4.set_aspect('equal')
ax4.axis('off')

# Add colorbar for panel 4
cbar = plt.colorbar(collection, ax=ax4, shrink=0.6, label='Min angle (degrees)')

plt.tight_layout()
plt.savefig('delaunay_concept_diagram.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("Concept diagram saved as: delaunay_concept_diagram.png")
