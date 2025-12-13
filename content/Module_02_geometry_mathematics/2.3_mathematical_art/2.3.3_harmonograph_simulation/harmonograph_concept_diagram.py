import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

# =============================================================================
# Create figure with three panels
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
fig.suptitle('How Two Pendulums Create Harmonograph Patterns', fontsize=14, fontweight='bold')

# Parameters for demonstration
t = np.linspace(0, 20, 1000)
freq_x, freq_y = 3, 2
amp = 1.0
damping = 0.05
decay = np.exp(-damping * t)

# Calculate x and y oscillations
x = amp * np.sin(freq_x * t) * decay
y = amp * np.sin(freq_y * t + np.pi/2) * decay

# =============================================================================
# Panel 1: X pendulum oscillation
# =============================================================================
ax1 = axes[0]
ax1.set_title('Pendulum X (Horizontal)\nfreq = 3, damped', fontsize=11, fontweight='bold')
ax1.plot(t, x, color='#ff6b6b', linewidth=2, label='x(t)')
ax1.fill_between(t, 0, x, alpha=0.3, color='#ff6b6b')
ax1.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')

# Add envelope
ax1.plot(t, decay, 'k--', alpha=0.5, linewidth=1, label='Decay envelope')
ax1.plot(t, -decay, 'k--', alpha=0.5, linewidth=1)

ax1.set_xlabel('Time (t)', fontsize=10)
ax1.set_ylabel('X Position', fontsize=10)
ax1.set_ylim(-1.2, 1.2)
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)

# Add equation annotation
ax1.annotate(r'$x(t) = A \cdot \sin(f_x \cdot t) \cdot e^{-dt}$',
             xy=(10, -0.9), fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# =============================================================================
# Panel 2: Y pendulum oscillation
# =============================================================================
ax2 = axes[1]
ax2.set_title('Pendulum Y (Vertical)\nfreq = 2, damped, phase-shifted', fontsize=11, fontweight='bold')
ax2.plot(t, y, color='#4ecdc4', linewidth=2, label='y(t)')
ax2.fill_between(t, 0, y, alpha=0.3, color='#4ecdc4')
ax2.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')

# Add envelope
ax2.plot(t, decay, 'k--', alpha=0.5, linewidth=1, label='Decay envelope')
ax2.plot(t, -decay, 'k--', alpha=0.5, linewidth=1)

ax2.set_xlabel('Time (t)', fontsize=10)
ax2.set_ylabel('Y Position', fontsize=10)
ax2.set_ylim(-1.2, 1.2)
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)

# Add equation annotation
ax2.annotate(r'$y(t) = A \cdot \sin(f_y \cdot t + \phi) \cdot e^{-dt}$',
             xy=(10, -0.9), fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# =============================================================================
# Panel 3: Combined pattern (X-Y parametric plot)
# =============================================================================
ax3 = axes[2]
ax3.set_title('Combined Pattern\n(X vs Y creates the harmonograph)', fontsize=11, fontweight='bold')

# Color the line by time for visual effect
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Use scatter with color gradient to show time evolution
colors = plt.cm.plasma(np.linspace(0, 1, len(t)))
ax3.scatter(x, y, c=colors, s=1, alpha=0.8)

# Draw arrows showing direction of travel
arrow_indices = [50, 200, 400]
for idx in arrow_indices:
    if idx + 5 < len(x):
        ax3.annotate('', xy=(x[idx+5], y[idx+5]), xytext=(x[idx], y[idx]),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

ax3.set_xlabel('X Position (Pendulum X)', fontsize=10)
ax3.set_ylabel('Y Position (Pendulum Y)', fontsize=10)
ax3.set_xlim(-1.2, 1.2)
ax3.set_ylim(-1.2, 1.2)
ax3.set_aspect('equal')
ax3.grid(True, alpha=0.3)

# Add annotation
ax3.annotate('Start (t=0)', xy=(x[0], y[0]), xytext=(0.5, 0.8),
             fontsize=9, arrowprops=dict(arrowstyle='->', color='green'),
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
ax3.annotate('End (damped)', xy=(x[-1], y[-1]), xytext=(-0.8, -0.8),
             fontsize=9, arrowprops=dict(arrowstyle='->', color='red'),
             bbox=dict(boxstyle='round', facecolor='lightsalmon', alpha=0.8))

# Add colorbar to show time
cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='plasma'), ax=ax3, orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label('Time (early to late)', fontsize=9)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(['Start', 'End'])

# =============================================================================
# Save figure
# =============================================================================
plt.tight_layout()
output_path = SCRIPT_DIR / 'harmonograph_concept_diagram.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

