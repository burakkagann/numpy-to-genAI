"""
Generate a visual diagram showing how the dragon curve sequence
builds recursively through iterations 0-4.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def invert_sequence(sequence):
    """Swap all L's and R's in the sequence."""
    return ''.join(['L' if char == 'R' else 'R' for char in sequence])


def generate_dragon_sequence(initial_turn, depth):
    """Generate the dragon curve turn sequence recursively."""
    if depth == 0:
        return initial_turn
    else:
        previous = generate_dragon_sequence(initial_turn, depth - 1)
        second_half = invert_sequence(previous[::-1])
        return previous + 'R' + second_half


# Create figure
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

# Title
ax.text(5, 5.7, 'Dragon Curve Sequence Construction',
        fontsize=16, fontweight='bold', ha='center', va='top')

# Show sequence for depths 0-4
depths = [0, 1, 2, 3, 4]
y_positions = [4.8, 3.8, 2.8, 1.8, 0.8]

for depth, y_pos in zip(depths, y_positions):
    seq = generate_dragon_sequence('R', depth)

    # Depth label
    ax.text(0.3, y_pos, f'D{depth} =', fontsize=12, fontweight='bold',
            ha='left', va='center', fontfamily='monospace')

    # Sequence with color coding
    x_start = 1.2
    char_width = 0.18

    # For depths > 0, highlight the structure
    if depth == 0:
        # Just R
        ax.text(x_start, y_pos, 'R', fontsize=11, ha='left', va='center',
                fontfamily='monospace', color='#2E86AB', fontweight='bold')
    else:
        # Get previous sequence for highlighting
        prev_len = len(generate_dragon_sequence('R', depth - 1))

        for i, char in enumerate(seq):
            # Color coding: first half (blue), middle R (red), second half (green)
            if i < prev_len:
                color = '#2E86AB'  # Blue for first half
            elif i == prev_len:
                color = '#E63946'  # Red for middle R (fold point)
            else:
                color = '#2A9D8F'  # Teal for second half

            ax.text(x_start + i * char_width, y_pos, char, fontsize=11,
                    ha='left', va='center', fontfamily='monospace',
                    color=color, fontweight='bold')

    # Length annotation
    length = len(seq)
    ax.text(9.5, y_pos, f'Length: {length}', fontsize=10, ha='right', va='center',
            color='gray', style='italic')

# Legend
legend_y = 0.15
ax.add_patch(mpatches.Rectangle((1, legend_y - 0.15), 0.3, 0.25,
                                  facecolor='#2E86AB', alpha=0.3))
ax.text(1.4, legend_y, 'Previous sequence', fontsize=9, va='center')

ax.add_patch(mpatches.Rectangle((4, legend_y - 0.15), 0.3, 0.25,
                                  facecolor='#E63946', alpha=0.3))
ax.text(4.4, legend_y, 'Middle fold (R)', fontsize=9, va='center')

ax.add_patch(mpatches.Rectangle((7, legend_y - 0.15), 0.3, 0.25,
                                  facecolor='#2A9D8F', alpha=0.3))
ax.text(7.4, legend_y, 'Reversed + inverted', fontsize=9, va='center')

# Formula annotation
formula_box = mpatches.FancyBboxPatch((0.3, 5.1), 9.4, 0.45,
                                        boxstyle='round,pad=0.05',
                                        facecolor='#f0f0f0', edgecolor='gray')
ax.add_patch(formula_box)
ax.text(5, 5.32, r'$D_n = D_{n-1}$ + R + reverse(invert($D_{n-1}$))',
        fontsize=11, ha='center', va='center', fontfamily='serif')

plt.tight_layout()
plt.savefig('recursive_sequence_visual.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("Saved recursive_sequence_visual.png")
