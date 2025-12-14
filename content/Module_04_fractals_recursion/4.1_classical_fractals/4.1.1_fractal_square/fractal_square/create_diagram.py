import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def create_recursion_diagram():
    """
    Create a diagram illustrating the fractal square recursion pattern.
    Shows the 3x3 grid division and highlights the four corner regions
    that get processed recursively.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=150)

    # Main square boundary
    main_size = 9

    # Draw the main square outline
    main_square = patches.Rectangle(
        (0, 0), main_size, main_size,
        linewidth=3, edgecolor='black', facecolor='none'
    )
    ax.add_patch(main_square)

    # Draw the 3x3 grid lines
    third = main_size / 3
    for i in range(1, 3):
        # Vertical lines
        ax.axvline(x=i * third, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        # Horizontal lines
        ax.axhline(y=i * third, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

    # Fill the center square (the part that gets colored)
    center_square = patches.Rectangle(
        (third, third), third, third,
        linewidth=2, edgecolor='darkgreen', facecolor='lightgreen', alpha=0.8
    )
    ax.add_patch(center_square)

    # Highlight the four corner regions that recurse
    corner_colors = ['#FFB3BA', '#BAFFC9', '#BAE1FF', '#FFFFBA']  # Pastel colors
    corner_labels = ['Corner 1\n(Recurse)', 'Corner 2\n(Recurse)',
                     'Corner 3\n(Recurse)', 'Corner 4\n(Recurse)']

    # Top-left corner (0,0 to 2/3, 2/3 of main)
    corners = [
        (0, 2*third, 2*third, third),           # Top-left
        (third, 2*third, 2*third, third),       # Top-right
        (0, 0, 2*third, third),                 # Bottom-left
        (third, 0, 2*third, third),             # Bottom-right
    ]

    # Draw corner indicators with arrows
    arrow_positions = [
        (third/2, 2*third + third/2, third/2, third/2),      # Top-left
        (2*third + third/2, 2*third + third/2, third/2, third/2),  # Top-right
        (third/2, third/2, third/2, third/2),                # Bottom-left
        (2*third + third/2, third/2, third/2, third/2),      # Bottom-right
    ]

    # Draw the recursive regions with distinct shading
    for i, ((x, y, w, h), color) in enumerate(zip(corners, corner_colors)):
        corner_rect = patches.FancyBboxPatch(
            (x + 0.1, y + 0.1), w - 0.2, h - 0.2,
            boxstyle="round,pad=0.05",
            linewidth=2, edgecolor='blue', facecolor=color, alpha=0.4
        )
        ax.add_patch(corner_rect)

    # Add labels
    ax.text(1.5*third, 1.5*third, 'Center\n(Fill)', fontsize=14,
            ha='center', va='center', fontweight='bold', color='darkgreen')

    ax.text(third/2, 2.5*third, '1', fontsize=20,
            ha='center', va='center', fontweight='bold', color='blue')
    ax.text(2.5*third, 2.5*third, '2', fontsize=20,
            ha='center', va='center', fontweight='bold', color='blue')
    ax.text(third/2, third/2, '3', fontsize=20,
            ha='center', va='center', fontweight='bold', color='blue')
    ax.text(2.5*third, third/2, '4', fontsize=20,
            ha='center', va='center', fontweight='bold', color='blue')

    # Add title and annotations
    ax.set_title('Fractal Square: Recursive Division Pattern', fontsize=16, fontweight='bold', pad=20)

    # Add legend
    legend_elements = [
        patches.Patch(facecolor='lightgreen', edgecolor='darkgreen', label='Center (filled each iteration)'),
        patches.Patch(facecolor='lightblue', edgecolor='blue', alpha=0.5, label='Corners (recursive calls)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, -0.05), fontsize=10)

    # Add step explanation
    step_text = "Algorithm Steps:\n1. Divide region into 3x3 grid\n2. Fill center square (green)\n3. Recurse on 4 corners (1-4)\n4. Stop when depth = 0"
    ax.text(main_size + 0.5, main_size/2, step_text, fontsize=11,
            va='center', ha='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Set axis properties
    ax.set_xlim(-0.5, main_size + 4)
    ax.set_ylim(-1.5, main_size + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Save the diagram
    plt.tight_layout()
    plt.savefig('recursion_diagram.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Saved recursion_diagram.png")


if __name__ == '__main__':
    create_recursion_diagram()
