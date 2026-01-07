"""
Generate Data Flow Diagram

Creates a visualization showing how data flows through a TouchDesigner
node network, illustrating the concept of "cooking" and dependencies.

This diagram demonstrates:
- Data moving through connected operators
- The dependency graph concept
- How changes propagate through the network
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
from matplotlib.collections import PatchCollection


def create_dataflow_diagram():
    """Create a data flow visualization for TD networks."""

    fig, ax = plt.subplots(figsize=(12, 6))

    # Colors
    bg_color = '#1a1a2e'
    box_color = '#16213e'
    text_color = '#ffffff'
    top_color = '#8b5cf6'      # Purple for TOPs
    chop_color = '#10b981'     # Green for CHOPs
    flow_color = '#f59e0b'     # Orange for data flow
    cook_color = '#ef4444'     # Red for cooking indicator

    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.set_xlim(0, 12)
    ax.set_ylim(2.5, 8.5)
    ax.axis('off')

    # Title
    ax.set_title('Data Flow in TouchDesigner Networks',
                 fontsize=18, fontweight='bold', color=text_color, pad=20)

    # =========================================
    # Network Layout
    # =========================================

    # Define nodes with positions
    nodes = {
        'noise': {'pos': (1.5, 7), 'label': 'Noise TOP', 'type': 'TOP',
                  'desc': 'Generates\nrandom pixels'},
        'level': {'pos': (4.5, 7), 'label': 'Level TOP', 'type': 'TOP',
                  'desc': 'Adjusts\nbrightness'},
        'blur': {'pos': (7.5, 7), 'label': 'Blur TOP', 'type': 'TOP',
                 'desc': 'Smooths\nimage'},
        'null': {'pos': (10.5, 7), 'label': 'Null TOP', 'type': 'TOP',
                 'desc': 'Output\nreference'},
        'lfo': {'pos': (4.5, 4), 'label': 'LFO CHOP', 'type': 'CHOP',
                'desc': 'Oscillating\nvalues'},
        'math': {'pos': (7.5, 4), 'label': 'Math CHOP', 'type': 'CHOP',
                 'desc': 'Scales\nvalues'},
    }

    type_colors = {'TOP': top_color, 'CHOP': chop_color}

    # Draw nodes
    for node_id, node in nodes.items():
        x, y = node['pos']
        color = type_colors[node['type']]

        # Main node box
        box = FancyBboxPatch((x - 0.9, y - 0.6), 1.8, 1.2,
                             boxstyle="round,pad=0.05,rounding_size=0.15",
                             facecolor=box_color, edgecolor=color, linewidth=2.5)
        ax.add_patch(box)

        # Node label
        ax.text(x, y + 0.25, node['label'], fontsize=9, fontweight='bold',
                color=text_color, ha='center', va='center')

        # Node type
        ax.text(x, y - 0.15, node['type'], fontsize=8,
                color=color, ha='center', va='center')

        # Description below node
        ax.text(x, y - 1.1, node['desc'], fontsize=7,
                color='#718096', ha='center', va='top', linespacing=1.2)

    # =========================================
    # Data Flow Arrows
    # =========================================

    # TOP chain (horizontal flow)
    top_connections = [
        ((2.4, 7), (3.6, 7)),      # noise -> level
        ((5.4, 7), (6.6, 7)),      # level -> blur
        ((8.4, 7), (9.6, 7)),      # blur -> null
    ]

    for start, end in top_connections:
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='-|>', color=flow_color, lw=3,
                                   mutation_scale=15))

    # CHOP chain (horizontal)
    ax.annotate('', xy=(6.6, 4), xytext=(5.4, 4),
                arrowprops=dict(arrowstyle='-|>', color=flow_color, lw=3,
                               mutation_scale=15))

    # CHOP -> TOP parameter connection (vertical, dashed style)
    ax.annotate('', xy=(7.5, 6.4), xytext=(7.5, 4.6),
                arrowprops=dict(arrowstyle='-|>', color=chop_color, lw=2,
                               linestyle='--', mutation_scale=12))

    # Label the parameter connection
    ax.text(8.2, 5.5, 'Parameter\nExport', fontsize=8, color=chop_color,
            ha='left', va='center', style='italic')

    # =========================================
    # Data Type Labels on Flow
    # =========================================

    # Image data flowing through TOPs
    ax.text(3, 7.5, '2D Image', fontsize=8, color=top_color,
            ha='center', bbox=dict(boxstyle='round', facecolor=bg_color,
                                   edgecolor=top_color, alpha=0.8))
    ax.text(6, 7.5, '2D Image', fontsize=8, color=top_color,
            ha='center', bbox=dict(boxstyle='round', facecolor=bg_color,
                                   edgecolor=top_color, alpha=0.8))
    ax.text(9, 7.5, '2D Image', fontsize=8, color=top_color,
            ha='center', bbox=dict(boxstyle='round', facecolor=bg_color,
                                   edgecolor=top_color, alpha=0.8))

    # Numeric data in CHOPs
    ax.text(6, 4.5, 'Number', fontsize=8, color=chop_color,
            ha='center', bbox=dict(boxstyle='round', facecolor=bg_color,
                                   edgecolor=chop_color, alpha=0.8))

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=top_color, edgecolor='white', label='TOP (Texture/Image Data)'),
        mpatches.Patch(facecolor=chop_color, edgecolor='white', label='CHOP (Channel/Numeric Data)'),
        mpatches.Patch(facecolor=flow_color, edgecolor='white', label='Data Flow Direction'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
              facecolor=box_color, edgecolor='white', labelcolor='white')

    plt.tight_layout()
    plt.savefig('dataflow_diagram.png', dpi=150, facecolor=bg_color,
                edgecolor='none', bbox_inches='tight')
    plt.close()

    print("Created: dataflow_diagram.png")


if __name__ == '__main__':
    create_dataflow_diagram()
