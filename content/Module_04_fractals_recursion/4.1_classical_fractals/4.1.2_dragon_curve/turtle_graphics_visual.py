"""
Generate a visual diagram showing the turtle graphics direction system
and how L/R turns work.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np


# Create figure with two panels
fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

# ==== Panel 1: Direction System ====
ax1 = axes[0]
ax1.set_xlim(-2, 2)
ax1.set_ylim(-2, 2)
ax1.set_aspect('equal')
ax1.axis('off')
ax1.set_title('Four Cardinal Directions', fontsize=14, fontweight='bold', pad=10)

# Direction arrows
arrow_props = dict(arrowstyle='->', color='#2E86AB', lw=2, mutation_scale=20)

# UP (index 1)
ax1.annotate('', xy=(0, 1.3), xytext=(0, 0.3),
             arrowprops=dict(arrowstyle='->', color='#E63946', lw=3, mutation_scale=25))
ax1.text(0, 1.6, 'UP (1)', ha='center', fontsize=11, fontweight='bold', color='#E63946')

# DOWN (index 3)
ax1.annotate('', xy=(0, -1.3), xytext=(0, -0.3),
             arrowprops=dict(arrowstyle='->', color='#457B9D', lw=3, mutation_scale=25))
ax1.text(0, -1.6, 'DOWN (3)', ha='center', fontsize=11, fontweight='bold', color='#457B9D')

# LEFT (index 0)
ax1.annotate('', xy=(-1.3, 0), xytext=(-0.3, 0),
             arrowprops=dict(arrowstyle='->', color='#2A9D8F', lw=3, mutation_scale=25))
ax1.text(-1.6, 0, 'LEFT (0)', ha='center', fontsize=11, fontweight='bold', color='#2A9D8F',
         rotation=90, va='center')

# RIGHT (index 2)
ax1.annotate('', xy=(1.3, 0), xytext=(0.3, 0),
             arrowprops=dict(arrowstyle='->', color='#F4A261', lw=3, mutation_scale=25))
ax1.text(1.6, 0, 'RIGHT (2)', ha='center', fontsize=11, fontweight='bold', color='#F4A261',
         rotation=270, va='center')

# Center dot (turtle position)
ax1.plot(0, 0, 'ko', markersize=10)
ax1.text(0, -0.15, 'Turtle', ha='center', va='top', fontsize=9, style='italic')

# ==== Panel 2: Turn Examples ====
ax2 = axes[1]
ax2.set_xlim(-0.5, 3.5)
ax2.set_ylim(-1.5, 2)
ax2.set_aspect('equal')
ax2.axis('off')
ax2.set_title('Turn Operations', fontsize=14, fontweight='bold', pad=10)

# Example 1: Turn Right from UP
# Starting facing UP
ax2.annotate('', xy=(0.5, 1.2), xytext=(0.5, 0.5),
             arrowprops=dict(arrowstyle='->', color='#E63946', lw=2, mutation_scale=15))
ax2.text(0.5, 1.4, 'UP', ha='center', fontsize=9, color='#E63946')

# Curved arrow showing rotation
theta = np.linspace(np.pi/2, 0, 20)
r = 0.3
x_curve = 0.5 + r * np.cos(theta)
y_curve = 0.5 + r * np.sin(theta)
ax2.plot(x_curve, y_curve, 'g--', lw=1.5)
ax2.annotate('', xy=(0.8, 0.5), xytext=(0.75, 0.7),
             arrowprops=dict(arrowstyle='->', color='green', lw=1.5, mutation_scale=10))

# After turning right: facing RIGHT
ax2.annotate('', xy=(1.2, 0.5), xytext=(0.5, 0.5),
             arrowprops=dict(arrowstyle='->', color='#F4A261', lw=2, mutation_scale=15))
ax2.text(1.3, 0.35, 'RIGHT', ha='left', fontsize=9, color='#F4A261')

ax2.text(0.85, -0.2, 'Turn R\n(+1 mod 4)', ha='center', fontsize=9, color='green',
         fontweight='bold')

# Example 2: Turn Left from UP
# Starting facing UP
ax2.annotate('', xy=(2.5, 1.2), xytext=(2.5, 0.5),
             arrowprops=dict(arrowstyle='->', color='#E63946', lw=2, mutation_scale=15))
ax2.text(2.5, 1.4, 'UP', ha='center', fontsize=9, color='#E63946')

# Curved arrow showing rotation (counter-clockwise)
theta = np.linspace(np.pi/2, np.pi, 20)
r = 0.3
x_curve = 2.5 + r * np.cos(theta)
y_curve = 0.5 + r * np.sin(theta)
ax2.plot(x_curve, y_curve, 'b--', lw=1.5)
ax2.annotate('', xy=(2.2, 0.5), xytext=(2.25, 0.7),
             arrowprops=dict(arrowstyle='->', color='blue', lw=1.5, mutation_scale=10))

# After turning left: facing LEFT
ax2.annotate('', xy=(1.8, 0.5), xytext=(2.5, 0.5),
             arrowprops=dict(arrowstyle='->', color='#2A9D8F', lw=2, mutation_scale=15))
ax2.text(1.7, 0.35, 'LEFT', ha='right', fontsize=9, color='#2A9D8F')

ax2.text(2.15, -0.2, 'Turn L\n(+3 mod 4)', ha='center', fontsize=9, color='blue',
         fontweight='bold')

# Code box at bottom
code_box = mpatches.FancyBboxPatch((-0.3, -1.4), 3.6, 0.7,
                                    boxstyle='round,pad=0.05',
                                    facecolor='#f8f8f8', edgecolor='gray')
ax2.add_patch(code_box)
ax2.text(1.5, -1.05, 'turn_right(d) = (d + 1) % 4\nturn_left(d) = (d + 3) % 4',
         ha='center', va='center', fontsize=10, fontfamily='monospace')

plt.tight_layout()
plt.savefig('turtle_graphics_visual.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("Saved turtle_graphics_visual.png")
