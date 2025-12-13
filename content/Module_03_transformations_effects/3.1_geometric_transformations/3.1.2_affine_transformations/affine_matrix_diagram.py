import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create figure with dark background
fig, ax = plt.subplots(figsize=(10, 8), facecolor='#1e1e2e')
ax.set_facecolor('#1e1e2e')

# Matrix cell dimensions
cell_width = 1.5
cell_height = 1.0
matrix_x = 2.5
matrix_y = 3.5

# Colors
primary_color = '#89b4fa'      # Blue for linear transform
secondary_color = '#fab387'    # Orange for translation
fixed_color = '#6c7086'        # Gray for fixed values
text_color = '#cdd6f4'         # Light text
accent_color = '#a6e3a1'       # Green for labels

# Draw matrix cells
for row in range(3):
    for col in range(3):
        x = matrix_x + col * cell_width
        y = matrix_y - row * cell_height

        # Determine cell color based on position
        if row < 2 and col < 2:
            # Linear transformation part (top-left 2x2)
            color = primary_color
            alpha = 0.3
        elif row < 2 and col == 2:
            # Translation part (right column, top 2 rows)
            color = secondary_color
            alpha = 0.3
        else:
            # Fixed values (bottom row)
            color = fixed_color
            alpha = 0.2

        # Draw cell
        rect = patches.FancyBboxPatch(
            (x, y), cell_width, cell_height,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=color, edgecolor=text_color,
            alpha=alpha, linewidth=2
        )
        ax.add_patch(rect)

# Matrix element labels
matrix_labels = [
    ['a', 'b', 'tx'],
    ['c', 'd', 'ty'],
    ['0', '0', '1']
]

label_colors = [
    [primary_color, primary_color, secondary_color],
    [primary_color, primary_color, secondary_color],
    [fixed_color, fixed_color, fixed_color]
]

for row in range(3):
    for col in range(3):
        x = matrix_x + col * cell_width + cell_width / 2
        y = matrix_y - row * cell_height + cell_height / 2
        ax.text(x, y, matrix_labels[row][col],
                fontsize=24, fontweight='bold',
                ha='center', va='center',
                color=label_colors[row][col])

# Draw brackets
bracket_width = 0.15
left_bracket_x = matrix_x - 0.3
right_bracket_x = matrix_x + 3 * cell_width + 0.15

# Left bracket
ax.plot([left_bracket_x + bracket_width, left_bracket_x, left_bracket_x, left_bracket_x + bracket_width],
        [matrix_y + cell_height + 0.1, matrix_y + cell_height + 0.1,
         matrix_y - 2 * cell_height - 0.1, matrix_y - 2 * cell_height - 0.1],
        color=text_color, linewidth=3)

# Right bracket
ax.plot([right_bracket_x - bracket_width, right_bracket_x, right_bracket_x, right_bracket_x - bracket_width],
        [matrix_y + cell_height + 0.1, matrix_y + cell_height + 0.1,
         matrix_y - 2 * cell_height - 0.1, matrix_y - 2 * cell_height - 0.1],
        color=text_color, linewidth=3)

# Title
ax.text(matrix_x + 1.5 * cell_width, matrix_y + cell_height + 0.7,
        '2D Affine Transformation Matrix',
        fontsize=20, fontweight='bold', ha='center', color=text_color)

# Annotations with arrows
annotation_style = dict(
    arrowstyle='->', color=text_color,
    connectionstyle='arc3,rad=0.2'
)

# Linear transformation annotation
ax.annotate('Linear Transform\n(scale, rotate, shear)',
            xy=(matrix_x + cell_width, matrix_y + 0.5),
            xytext=(0.5, matrix_y + 0.5),
            fontsize=12, ha='center', color=primary_color,
            arrowprops=dict(arrowstyle='->', color=primary_color, lw=2))

# Translation annotation
ax.annotate('Translation\n(move x, y)',
            xy=(matrix_x + 2.5 * cell_width, matrix_y + 0.5),
            xytext=(matrix_x + 4.5 * cell_width, matrix_y + 0.5),
            fontsize=12, ha='center', color=secondary_color,
            arrowprops=dict(arrowstyle='->', color=secondary_color, lw=2))

# Fixed values annotation
ax.annotate('Fixed values\n(homogeneous coords)',
            xy=(matrix_x + 1.5 * cell_width, matrix_y - 2 * cell_height),
            xytext=(matrix_x + 1.5 * cell_width, matrix_y - 3.5 * cell_height),
            fontsize=12, ha='center', color=fixed_color,
            arrowprops=dict(arrowstyle='->', color=fixed_color, lw=2))

# Equation at bottom (simplified for matplotlib compatibility)
eq_y = 0.8
ax.text(matrix_x + 1.5 * cell_width, eq_y,
        r"$x' = a \cdot x + b \cdot y + t_x$   |   $y' = c \cdot x + d \cdot y + t_y$",
        fontsize=14, ha='center', va='center', color=text_color)

# Set axis limits and remove axes
ax.set_xlim(-0.5, 10)
ax.set_ylim(-0.5, 6)
ax.set_aspect('equal')
ax.axis('off')

# Save figure
plt.tight_layout()
plt.savefig('affine_matrix_diagram.png', dpi=150, facecolor='#1e1e2e',
            bbox_inches='tight', pad_inches=0.3)
plt.close()
