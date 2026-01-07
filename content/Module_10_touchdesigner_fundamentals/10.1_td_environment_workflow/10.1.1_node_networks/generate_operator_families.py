"""
Generate Operator Families Diagram

Creates a four-quadrant visualization of TouchDesigner's operator families:
- TOP (Texture Operators): Image and video processing
- CHOP (Channel Operators): Numeric data and animation
- SOP (Surface Operators): 3D geometry
- DAT (Data Operators): Text, tables, and scripts

This diagram is crucial for learners to understand the "type system"
of TouchDesigner, as different operators handle different types of data.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle


def create_operator_families_diagram():
    """Create a four-quadrant operator family visualization."""

    fig, ax = plt.subplots(figsize=(10, 10))

    # Colors
    bg_color = '#1a1a2e'
    box_color = '#16213e'
    text_color = '#ffffff'

    # Operator family colors
    top_color = '#8b5cf6'      # Purple - Textures/Images
    chop_color = '#10b981'     # Green - Channels/Data
    sop_color = '#f59e0b'      # Orange - Surfaces/3D
    dat_color = '#3b82f6'      # Blue - Data/Scripts

    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.set_title('The Four Operator Families',
                 fontsize=20, fontweight='bold', color=text_color, pad=20)

    # =========================================
    # Quadrant Layout
    # =========================================

    quadrants = [
        {
            'name': 'TOP',
            'full_name': 'Texture Operators',
            'color': top_color,
            'pos': (2.5, 7.5),  # Upper left
            'data_type': '2D Images & Video',
            'examples': ['Noise', 'Blur', 'Level', 'Composite'],
            'numpy_equiv': 'NumPy 2D/3D Arrays!',
            'icon_type': 'image'
        },
        {
            'name': 'CHOP',
            'full_name': 'Channel Operators',
            'color': chop_color,
            'pos': (7.5, 7.5),  # Upper right
            'data_type': 'Numbers & Curves',
            'examples': ['LFO', 'Math', 'Filter', 'Audio'],
            'numpy_equiv': '1D Arrays / Time Series',
            'icon_type': 'wave'
        },
        {
            'name': 'SOP',
            'full_name': 'Surface Operators',
            'color': sop_color,
            'pos': (2.5, 2.5),  # Lower left
            'data_type': '3D Geometry',
            'examples': ['Sphere', 'Box', 'Transform', 'Noise'],
            'numpy_equiv': 'Point Clouds / Meshes',
            'icon_type': 'cube'
        },
        {
            'name': 'DAT',
            'full_name': 'Data Operators',
            'color': dat_color,
            'pos': (7.5, 2.5),  # Lower right
            'data_type': 'Text, Tables, Scripts',
            'examples': ['Table', 'Text', 'Script', 'Web'],
            'numpy_equiv': 'DataFrames / Strings',
            'icon_type': 'table'
        },
    ]

    for quad in quadrants:
        x, y = quad['pos']
        color = quad['color']

        # Quadrant background box
        box = FancyBboxPatch((x - 2.2, y - 2.2), 4.4, 4.4,
                             boxstyle="round,pad=0.1,rounding_size=0.3",
                             facecolor=box_color, edgecolor=color,
                             linewidth=3, alpha=0.95)
        ax.add_patch(box)

        # Operator name (large)
        ax.text(x, y + 1.5, quad['name'], fontsize=28, fontweight='bold',
                color=color, ha='center', va='center')

        # Full name
        ax.text(x, y + 0.9, quad['full_name'], fontsize=11,
                color=text_color, ha='center', va='center')

        # Data type
        ax.text(x, y + 0.3, f"Data: {quad['data_type']}", fontsize=9,
                color='#a0aec0', ha='center', va='center')

        # Horizontal line separator
        ax.plot([x - 1.8, x + 1.8], [y - 0.1, y - 0.1], color=color,
                linewidth=1, alpha=0.5)

        # Examples
        examples_text = ' | '.join(quad['examples'])
        ax.text(x, y - 0.5, 'Examples:', fontsize=8, color='#718096',
                ha='center', va='center')
        ax.text(x, y - 0.85, examples_text, fontsize=8, color=text_color,
                ha='center', va='center')

        # NumPy equivalent (highlighted for TOP)
        numpy_text = quad['numpy_equiv']
        if quad['name'] == 'TOP':
            # Highlight TOP's connection to NumPy
            highlight_box = FancyBboxPatch((x - 1.6, y - 1.65), 3.2, 0.5,
                                           boxstyle="round,pad=0.05,rounding_size=0.1",
                                           facecolor='#2d3748', edgecolor='#48bb78',
                                           linewidth=2)
            ax.add_patch(highlight_box)
            ax.text(x, y - 1.4, numpy_text, fontsize=9, fontweight='bold',
                    color='#48bb78', ha='center', va='center')
        else:
            ax.text(x, y - 1.4, numpy_text, fontsize=8,
                    color='#718096', ha='center', va='center', style='italic')

    # =========================================
    # Center Connection Indicator
    # =========================================

    # Central circle showing interconnection
    center = Circle((5, 5), 0.6, facecolor=bg_color, edgecolor=text_color,
                    linewidth=2)
    ax.add_patch(center)
    ax.text(5, 5, 'All\nConnect', fontsize=8, color=text_color,
            ha='center', va='center', fontweight='bold')

    # Lines from center to each quadrant
    for quad in quadrants:
        x, y = quad['pos']
        # Calculate direction to center
        dx = 5 - x
        dy = 5 - y
        length = np.sqrt(dx**2 + dy**2)
        # Start point (edge of quadrant box)
        start_x = x + (dx / length) * 1.8
        start_y = y + (dy / length) * 1.8
        # End point (edge of center circle)
        end_x = 5 - (dx / length) * 0.65
        end_y = 5 - (dy / length) * 0.65

        ax.plot([start_x, end_x], [start_y, end_y], color='#4a5568',
                linewidth=1.5, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('operator_families_diagram.png', dpi=150, facecolor=bg_color,
                edgecolor='none', bbox_inches='tight')
    plt.close()

    print("Created: operator_families_diagram.png")


if __name__ == '__main__':
    create_operator_families_diagram()
