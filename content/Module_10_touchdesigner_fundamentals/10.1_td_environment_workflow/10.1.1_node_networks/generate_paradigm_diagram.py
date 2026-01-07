"""
Generate Paradigm Comparison Diagram

Creates a visual comparison between sequential Python execution
and node-based TouchDesigner dataflow programming.

This diagram helps learners understand the fundamental paradigm shift
from linear code execution to parallel node evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as path_effects


def create_paradigm_diagram():
    """Create a side-by-side comparison of Python vs Node-based paradigms."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor('#1a1a2e')

    # Colors
    bg_color = '#1a1a2e'
    box_color = '#16213e'
    python_color = '#3776ab'  # Python blue
    td_color = '#ff6b35'      # TouchDesigner orange
    text_color = '#ffffff'
    arrow_color = '#4a5568'
    highlight_color = '#48bb78'

    # =========================================
    # LEFT SIDE: Sequential Python Execution
    # =========================================
    ax1.set_facecolor(bg_color)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Sequential Python Execution', fontsize=16, fontweight='bold',
                  color=python_color, pad=20)

    # Python code blocks (executed top to bottom)
    python_steps = [
        ('1. Load Image', 'img = load("photo.jpg")', 8.5),
        ('2. Convert', 'gray = to_grayscale(img)', 6.5),
        ('3. Blur', 'blurred = blur(gray, 5)', 4.5),
        ('4. Save', 'save(blurred, "out.jpg")', 2.5),
    ]

    for label, code, y_pos in python_steps:
        # Step box
        box = FancyBboxPatch((1, y_pos - 0.6), 8, 1.2,
                             boxstyle="round,pad=0.05,rounding_size=0.2",
                             facecolor=box_color, edgecolor=python_color, linewidth=2)
        ax1.add_patch(box)

        # Step number and label
        ax1.text(1.3, y_pos + 0.2, label, fontsize=10, fontweight='bold',
                color=python_color, va='center')

        # Code text
        ax1.text(1.3, y_pos - 0.2, code, fontsize=9, fontfamily='monospace',
                color=text_color, va='center')

    # Arrows between steps (sequential flow)
    for i in range(3):
        y_start = python_steps[i][2] - 0.6
        y_end = python_steps[i+1][2] + 0.6
        ax1.annotate('', xy=(5, y_end), xytext=(5, y_start),
                    arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2))

    # Execution indicator
    ax1.text(5, 0.8, 'Executes ONCE, top to bottom', fontsize=10,
             color='#a0aec0', ha='center', style='italic')

    # =========================================
    # RIGHT SIDE: Node-Based TD Execution
    # =========================================
    ax2.set_facecolor(bg_color)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('Node-Based TouchDesigner', fontsize=16, fontweight='bold',
                  color=td_color, pad=20)

    # TD nodes (connected network, not strictly linear)
    td_nodes = [
        ('Movie File In', 2, 7.5, 'TOP'),
        ('Monochrome', 5, 7.5, 'TOP'),
        ('Blur', 8, 7.5, 'TOP'),
        ('LFO', 2, 4.5, 'CHOP'),
        ('Null (Output)', 8, 4.5, 'TOP'),
    ]

    node_colors = {
        'TOP': '#8b5cf6',   # Purple for texture ops
        'CHOP': '#10b981',  # Green for channel ops
    }

    for name, x, y, op_type in td_nodes:
        # Node box
        color = node_colors.get(op_type, td_color)
        box = FancyBboxPatch((x - 1.2, y - 0.5), 2.4, 1.0,
                             boxstyle="round,pad=0.05,rounding_size=0.15",
                             facecolor=box_color, edgecolor=color, linewidth=2)
        ax2.add_patch(box)

        # Node name
        ax2.text(x, y + 0.15, name, fontsize=9, fontweight='bold',
                color=text_color, ha='center', va='center')

        # Operator type badge
        ax2.text(x, y - 0.25, op_type, fontsize=8,
                color=color, ha='center', va='center')

    # Connections (wires between nodes)
    connections = [
        ((3.2, 7.5), (3.8, 7.5)),    # Movie -> Mono
        ((6.2, 7.5), (6.8, 7.5)),    # Mono -> Blur
        ((8, 6.5), (8, 5.5)),        # Blur -> Null (vertical)
        ((3.2, 4.5), (6.8, 4.5)),    # LFO -> Null (parameter)
    ]

    for start, end in connections:
        ax2.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color=highlight_color, lw=2,
                                   connectionstyle="arc3,rad=0"))

    # Label for parameter connection
    ax2.text(5, 4.1, 'controls blur amount', fontsize=8, color='#a0aec0',
             ha='center', style='italic')

    # Execution indicator
    ax2.text(5, 2.5, 'Evaluates CONTINUOUSLY', fontsize=11,
             color=highlight_color, ha='center', fontweight='bold')
    ax2.text(5, 1.8, 'Every frame (60+ times/second)', fontsize=10,
             color='#a0aec0', ha='center', style='italic')
    ax2.text(5, 1.1, 'Changes propagate instantly!', fontsize=10,
             color='#a0aec0', ha='center', style='italic')

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='#8b5cf6', edgecolor='white', label='TOP (Texture/Image)'),
        mpatches.Patch(facecolor='#10b981', edgecolor='white', label='CHOP (Channel/Data)'),
    ]
    ax2.legend(handles=legend_elements, loc='upper left', fontsize=9,
               facecolor=box_color, edgecolor='white', labelcolor='white')

    # Main title
    fig.suptitle('Programming Paradigms: Sequential vs Node-Based',
                 fontsize=18, fontweight='bold', color=text_color, y=0.98)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig('paradigm_comparison.png', dpi=150, facecolor=bg_color,
                edgecolor='none', bbox_inches='tight')
    plt.close()

    print("Created: paradigm_comparison.png")


if __name__ == '__main__':
    create_paradigm_diagram()
