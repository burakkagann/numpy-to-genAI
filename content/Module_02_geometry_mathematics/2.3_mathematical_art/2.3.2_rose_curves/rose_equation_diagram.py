import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent

# Create figure with polar subplot
fig = plt.figure(figsize=(10, 8), dpi=150, facecolor='white')

# Main polar plot for the rose curve
ax = fig.add_subplot(111, projection='polar')
ax.set_facecolor('#f8f9fa')

# Rose curve parameters for demonstration
K_PARAMETER = 5
AMPLITUDE = 1.0

# Generate rose curve
theta = np.linspace(0, 2 * np.pi, 1000)
r = AMPLITUDE * np.cos(K_PARAMETER * theta)

# Plot the rose curve
ax.plot(theta, r, color='#ff6b9d', linewidth=2.5, label='Rose curve')

# Mark a specific point to show coordinates
highlight_theta = np.pi / 10  # 18 degrees
highlight_r = AMPLITUDE * np.cos(K_PARAMETER * highlight_theta)

# Draw radius line from center to point
ax.plot([0, highlight_theta], [0, highlight_r], 'b-', linewidth=2, alpha=0.8)
ax.plot(highlight_theta, highlight_r, 'bo', markersize=10, zorder=5)

# Add angle arc
arc_theta = np.linspace(0, highlight_theta, 20)
arc_r = np.full_like(arc_theta, 0.2)
ax.plot(arc_theta, arc_r, 'g-', linewidth=2, alpha=0.8)

# Annotations
ax.annotate(r'$\theta$', xy=(highlight_theta/2, 0.15), fontsize=14, color='green',
            ha='center', va='center')
ax.annotate(r'$r = a \cdot \cos(k\theta)$', xy=(highlight_theta + 0.2, highlight_r - 0.1),
            fontsize=12, color='blue', ha='left')

# Title and equation
ax.set_title('Rose Curve: $r = a \\cdot \\cos(k\\theta)$\n' +
             f'Parameters: $k = {K_PARAMETER}$, $a = {AMPLITUDE}$',
             fontsize=14, fontweight='bold', pad=20)

# Configure polar plot appearance
ax.set_rticks([0.25, 0.5, 0.75, 1.0])
ax.set_rlabel_position(45)
ax.grid(True, alpha=0.3)

# Add text box with key information
textstr = '\n'.join([
    'Key Properties:',
    f'• k = {K_PARAMETER} (odd) → {K_PARAMETER} petals',
    '• Even k → 2k petals',
    '• a = amplitude (petal length)'
])
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom', bbox=props)

plt.tight_layout()

# Save the diagram
output_path = SCRIPT_DIR / 'rose_equation_diagram.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
