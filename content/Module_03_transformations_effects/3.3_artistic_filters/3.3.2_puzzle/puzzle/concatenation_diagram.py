"""
Concatenation Diagram - Visual Explanation of vstack vs hstack

This script creates a conceptual diagram showing how np.vstack and np.hstack
combine arrays, with clear annotations about dimension requirements.

Author: Claude (NumPy-to-GenAI Project)
Date: 2025-01-20

Thesis Metadata:
    Framework: F1-Hands-On
    Module: Module_03_transformations_effects
    Exercise Type: Conceptual Diagram
    Cognitive Load: Medium
    New Concepts: Visual representation of array concatenation
    Prerequisites: Understanding of array shapes
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Array Concatenation: vstack vs hstack', fontsize=16, fontweight='bold')

# Colors for arrays
color_a = '#FF6B6B'  # Red
color_b = '#4ECDC4'  # Teal
color_result = '#95E1D3'  # Light green

# === LEFT SUBPLOT: vstack (Vertical Stacking) ===
ax1 = axes[0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.set_aspect('equal')
ax1.axis('off')
ax1.set_title('np.vstack([A, B])\nStacks Vertically (Top to Bottom)', fontsize=12, fontweight='bold')

# Array A (top)
rect_a = patches.Rectangle((1, 7), 3, 2, linewidth=2, edgecolor='black', facecolor=color_a)
ax1.add_patch(rect_a)
ax1.text(2.5, 8, 'A', fontsize=14, ha='center', va='center', fontweight='bold')
ax1.text(4.5, 8, 'Shape: (100, 200, 3)', fontsize=10, ha='left', va='center')

# Array B (below)
rect_b = patches.Rectangle((1, 4.5), 3, 2, linewidth=2, edgecolor='black', facecolor=color_b)
ax1.add_patch(rect_b)
ax1.text(2.5, 5.5, 'B', fontsize=14, ha='center', va='center', fontweight='bold')
ax1.text(4.5, 5.5, 'Shape: (150, 200, 3)', fontsize=10, ha='left', va='center')

# Arrow
ax1.annotate('', xy=(2.5, 3.5), xytext=(2.5, 4.3),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# Result (combined)
rect_result = patches.Rectangle((1, 0.5), 3, 2.5, linewidth=2, edgecolor='black', facecolor=color_result)
ax1.add_patch(rect_result)
ax1.text(2.5, 1.75, 'Result', fontsize=14, ha='center', va='center', fontweight='bold')
ax1.text(4.5, 1.75, 'Shape: (250, 200, 3)', fontsize=10, ha='left', va='center')

# Dimension annotation
ax1.text(5, 9.5, 'Widths must match!', fontsize=11, ha='left', va='center',
         style='italic', color='darkred', fontweight='bold')
ax1.annotate('', xy=(4.2, 8), xytext=(4.2, 5.5),
            arrowprops=dict(arrowstyle='<->', lw=1.5, color='darkred'))

# === RIGHT SUBPLOT: hstack (Horizontal Stacking) ===
ax2 = axes[1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.set_aspect('equal')
ax2.axis('off')
ax2.set_title('np.hstack([A, B])\nStacks Horizontally (Left to Right)', fontsize=12, fontweight='bold')

# Array A (left)
rect_a2 = patches.Rectangle((1, 6), 2, 3, linewidth=2, edgecolor='black', facecolor=color_a)
ax2.add_patch(rect_a2)
ax2.text(2, 7.5, 'A', fontsize=14, ha='center', va='center', fontweight='bold')
ax2.text(1, 5.5, 'Shape:\n(200, 100, 3)', fontsize=9, ha='center', va='top')

# Array B (right)
rect_b2 = patches.Rectangle((3.5, 6), 3, 3, linewidth=2, edgecolor='black', facecolor=color_b)
ax2.add_patch(rect_b2)
ax2.text(5, 7.5, 'B', fontsize=14, ha='center', va='center', fontweight='bold')
ax2.text(5, 5.5, 'Shape:\n(200, 150, 3)', fontsize=9, ha='center', va='top')

# Arrow
ax2.annotate('', xy=(4, 4.5), xytext=(4, 5.3),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# Result (combined)
rect_result2 = patches.Rectangle((1, 1), 5.5, 3, linewidth=2, edgecolor='black', facecolor=color_result)
ax2.add_patch(rect_result2)
ax2.text(3.75, 2.5, 'Result', fontsize=14, ha='center', va='center', fontweight='bold')
ax2.text(3.75, 0.5, 'Shape: (200, 250, 3)', fontsize=10, ha='center', va='top')

# Dimension annotation
ax2.text(1, 9.5, 'Heights must match!', fontsize=11, ha='left', va='center',
         style='italic', color='darkgreen', fontweight='bold')
ax2.annotate('', xy=(0.7, 6), xytext=(0.7, 9),
            arrowprops=dict(arrowstyle='<->', lw=1.5, color='darkgreen'))
ax2.annotate('', xy=(6.8, 6), xytext=(6.8, 9),
            arrowprops=dict(arrowstyle='<->', lw=1.5, color='darkgreen'))

plt.tight_layout()
plt.savefig('concatenation_diagram.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("Created concatenation_diagram.png - visual explanation of vstack vs hstack")
