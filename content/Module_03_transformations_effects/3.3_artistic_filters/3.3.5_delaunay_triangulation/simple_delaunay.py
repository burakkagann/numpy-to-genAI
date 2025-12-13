"""
Exercise 3.3.5: Simple Delaunay Triangulation

Generate a triangular mesh from random points using scipy.spatial.Delaunay.
This demonstrates the fundamental concept of Delaunay triangulation where
no point lies inside any triangle's circumcircle.

Author: Claude (NumPy-to-GenAI Project)
Date: 2025-12-02

Thesis Metadata:
    Framework: F1-Hands-On
    Module: Module_03_transformations_effects
    Exercise Type: Execute
    Cognitive Load: Low
    New Concepts: Delaunay triangulation, triangular mesh, circumcircle property
    Prerequisites: NumPy arrays, matplotlib basics

Research Question Contributions:
    RQ1 (Framework Design): Visual-first approach showing output before theory
    RQ2 (Cognitive Load): Single concept introduction with minimal code
    RQ4 (Assessment): Technical execution, visual verification
    RQ5 (Transfer): Foundation for mesh generation, computational geometry
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# Step 1: Generate random points in 2D space
# Using a fixed seed ensures reproducible results for learning
np.random.seed(42)
num_points = 50
points = np.random.rand(num_points, 2) * 400  # 50 points in 400x400 space

# Step 2: Compute Delaunay triangulation
# This creates triangles that maximize the minimum angle (no thin slivers)
triangulation = Delaunay(points)

# Step 3: Visualize the triangulation
plt.figure(figsize=(8, 8))
plt.triplot(points[:, 0], points[:, 1], triangulation.simplices,
            color='steelblue', linewidth=0.8)
plt.plot(points[:, 0], points[:, 1], 'o', color='coral', markersize=6)
plt.title(f'Delaunay Triangulation ({num_points} points, {len(triangulation.simplices)} triangles)')
plt.axis('equal')
plt.axis('off')
plt.tight_layout()
plt.savefig('simple_delaunay.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("Delaunay triangulation complete!")
print(f"Number of points: {num_points}")
print(f"Number of triangles: {len(triangulation.simplices)}")
print("Output saved as: simple_delaunay.png")
