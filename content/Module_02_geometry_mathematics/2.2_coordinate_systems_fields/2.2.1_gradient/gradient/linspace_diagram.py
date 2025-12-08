# Diagram generated with Claude Code
# Visualizes np.linspace() for educational purposes

import numpy as np
import matplotlib.pyplot as plt

# Create figure with increased height for better spacing
fig, axes = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [1.2, 2]})

# Top subplot: Number line visualization
ax1 = axes[0]
values = np.linspace(0, 255, 9)  # 9 points for clear visualization

# Draw number line
ax1.axhline(y=0, color='black', linewidth=2)
ax1.plot(values, [0]*len(values), 'o', color='#2196F3', markersize=12, zorder=5)

# Add value labels (increased offset for better spacing)
for i, v in enumerate(values):
    ax1.annotate(f'{int(v)}', (v, 0), textcoords="offset points",
                 xytext=(0, 20), ha='center', fontsize=10, fontweight='bold')

# Add position labels below (increased offset for better spacing)
positions = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
for i, (v, p) in enumerate(zip(values, positions)):
    ax1.annotate(f'[{p}]', (v, 0), textcoords="offset points",
                 xytext=(0, -25), ha='center', fontsize=9, color='gray')

ax1.set_xlim(-20, 275)
ax1.set_ylim(-1.0, 1.0)  # Expanded for better text spacing
ax1.set_title('np.linspace(0, 255, 9) generates 9 evenly spaced values',
              fontsize=12, fontweight='bold', pad=10)
ax1.axis('off')

# Add arrows at ends
ax1.annotate('', xy=(270, 0), xytext=(255, 0),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))
ax1.annotate('', xy=(-15, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))

# Bottom subplot: Show how it maps to gradient
ax2 = axes[1]

# Create a gradient bar
gradient = np.linspace(0, 255, 256).reshape(1, -1)
ax2.imshow(gradient, cmap='gray', aspect='auto', extent=[0, 255, 0, 1])

# Mark the 9 sample points
for i, v in enumerate(values):
    ax2.axvline(x=v, color='#FF5722', linewidth=1.5, linestyle='--', alpha=0.7)
    ax2.plot(v, 0.5, 'o', color='#FF5722', markersize=8, zorder=5)

# Add annotation
ax2.set_xlabel('Pixel value (0 = black, 255 = white)', fontsize=11)
ax2.set_title('Each value maps to a shade of gray in the gradient',
              fontsize=12, fontweight='bold', pad=10)
ax2.set_yticks([])
ax2.set_xlim(0, 255)

# Add formula annotation
fig.text(0.5, 0.02,
         'Formula: value[i] = start + i * (stop - start) / (num - 1)',
         ha='center', fontsize=10, style='italic', color='#666666')

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)  # Increased for formula text
plt.savefig('linspace_diagram.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

