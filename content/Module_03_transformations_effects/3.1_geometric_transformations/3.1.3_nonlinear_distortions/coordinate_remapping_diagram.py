import numpy as np
import matplotlib.pyplot as plt

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=100)

# Parameters
grid_size = 10
size = 100
amplitude = 8
frequency = 2

# Draw original grid (left panel)
ax1.set_xlim(0, size)
ax1.set_ylim(0, size)
ax1.set_aspect('equal')
ax1.set_title('Original Grid', fontsize=12, fontweight='bold')

# Draw grid lines
for i in range(grid_size + 1):
    pos = i * (size / grid_size)
    ax1.axhline(y=pos, color='blue', linewidth=1, alpha=0.7)
    ax1.axvline(x=pos, color='blue', linewidth=1, alpha=0.7)

ax1.set_xlabel('X coordinate')
ax1.set_ylabel('Y coordinate')

# Draw distorted grid (right panel)
ax2.set_xlim(0, size)
ax2.set_ylim(0, size)
ax2.set_aspect('equal')
ax2.set_title('After Wave Distortion', fontsize=12, fontweight='bold')

# Draw horizontal lines (unchanged)
for i in range(grid_size + 1):
    pos = i * (size / grid_size)
    ax2.axhline(y=pos, color='blue', linewidth=1, alpha=0.7)

# Draw distorted vertical lines
for i in range(grid_size + 1):
    x_base = i * (size / grid_size)
    y_vals = np.linspace(0, size, 50)
    x_vals = x_base + amplitude * np.sin(2 * np.pi * frequency * y_vals / size)
    ax2.plot(x_vals, y_vals, color='red', linewidth=1.5)

ax2.set_xlabel('X coordinate')
ax2.set_ylabel('Y coordinate')

# Add arrows showing remapping concept
ax2.annotate('', xy=(50, 75), xytext=(50 + amplitude * np.sin(2 * np.pi * frequency * 75 / size), 75),
             arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax2.text(55, 80, 'Shift', fontsize=9, color='green', fontweight='bold')

plt.tight_layout()
plt.savefig('coordinate_remapping_diagram.png', dpi=100, bbox_inches='tight')
plt.close()

