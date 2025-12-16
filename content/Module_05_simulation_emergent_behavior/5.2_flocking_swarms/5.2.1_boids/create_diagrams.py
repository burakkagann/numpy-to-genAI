import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Style settings
plt.style.use('dark_background')
BOID_COLOR = '#00C8C8'  # Cyan/teal
NEIGHBOR_COLOR = '#808080'  # Gray for neighbors
ARROW_COLOR = '#FF6B6B'  # Red-coral for force arrows
PERCEPTION_COLOR = '#404060'  # Dim circle for perception radius


def draw_boid(ax, x, y, angle, color=BOID_COLOR, size=0.15):
    """Draw a triangle representing a boid."""
    # Triangle pointing in direction of angle
    triangle = plt.Polygon([
        [x + np.cos(angle) * size, y + np.sin(angle) * size],
        [x + np.cos(angle + 2.5) * size * 0.6, y + np.sin(angle + 2.5) * size * 0.6],
        [x + np.cos(angle - 2.5) * size * 0.6, y + np.sin(angle - 2.5) * size * 0.6]
    ], color=color)
    ax.add_patch(triangle)


def create_separation_diagram():
    """Create diagram showing separation rule."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw perception radius
    perception = plt.Circle((0, 0), 1.0, fill=False, color=PERCEPTION_COLOR,
                           linestyle='--', linewidth=1.5, alpha=0.5)
    ax.add_patch(perception)

    # Central boid
    draw_boid(ax, 0, 0, np.pi/4, BOID_COLOR, 0.2)

    # Nearby boids (too close)
    neighbors = [
        (0.3, 0.2, np.pi/3),
        (-0.25, 0.15, np.pi/6),
        (0.1, -0.3, -np.pi/4),
    ]

    for nx, ny, angle in neighbors:
        draw_boid(ax, nx, ny, angle, NEIGHBOR_COLOR, 0.15)
        # Draw arrow pointing away from neighbor
        ax.annotate('', xy=(-nx * 0.8, -ny * 0.8), xytext=(0, 0),
                   arrowprops=dict(arrowstyle='->', color=ARROW_COLOR,
                                  lw=2, mutation_scale=15))

    ax.set_title('Separation: Steer Away from Nearby Boids',
                fontsize=14, fontweight='bold', color='white', pad=20)

    # Save
    output_path = os.path.join(SCRIPT_DIR, 'separation_diagram.png')
    plt.savefig(output_path, bbox_inches='tight', facecolor='#141420',
                edgecolor='none', dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def create_alignment_diagram():
    """Create diagram showing alignment rule."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw perception radius
    perception = plt.Circle((0, 0), 1.0, fill=False, color=PERCEPTION_COLOR,
                           linestyle='--', linewidth=1.5, alpha=0.5)
    ax.add_patch(perception)

    # Central boid pointing left
    draw_boid(ax, 0, 0, np.pi, BOID_COLOR, 0.2)

    # Neighbors all pointing same direction (up-right)
    avg_angle = np.pi/4
    neighbors = [
        (0.5, 0.3, avg_angle),
        (-0.4, 0.5, avg_angle + 0.1),
        (0.3, -0.4, avg_angle - 0.1),
        (-0.5, -0.2, avg_angle + 0.15),
    ]

    for nx, ny, angle in neighbors:
        draw_boid(ax, nx, ny, angle, NEIGHBOR_COLOR, 0.15)

    # Draw arrow showing desired alignment direction
    ax.annotate('', xy=(0.6, 0.6), xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', color=ARROW_COLOR,
                              lw=3, mutation_scale=20))

    # Label showing average direction
    ax.text(0.7, 0.7, 'Average\nHeading', fontsize=10, color=ARROW_COLOR,
           ha='left', va='bottom')

    ax.set_title('Alignment: Match Neighbors\' Heading',
                fontsize=14, fontweight='bold', color='white', pad=20)

    # Save
    output_path = os.path.join(SCRIPT_DIR, 'alignment_diagram.png')
    plt.savefig(output_path, bbox_inches='tight', facecolor='#141420',
                edgecolor='none', dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def create_cohesion_diagram():
    """Create diagram showing cohesion rule."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw perception radius
    perception = plt.Circle((0, 0), 1.0, fill=False, color=PERCEPTION_COLOR,
                           linestyle='--', linewidth=1.5, alpha=0.5)
    ax.add_patch(perception)

    # Central boid at edge
    draw_boid(ax, -0.6, -0.4, np.pi/4, BOID_COLOR, 0.2)

    # Neighbors clustered together
    neighbors = [
        (0.4, 0.3, np.pi/6),
        (0.5, 0.5, np.pi/4),
        (0.3, 0.6, np.pi/3),
        (0.6, 0.4, np.pi/5),
    ]

    for nx, ny, angle in neighbors:
        draw_boid(ax, nx, ny, angle, NEIGHBOR_COLOR, 0.15)

    # Calculate center of neighbors
    center_x = np.mean([n[0] for n in neighbors])
    center_y = np.mean([n[1] for n in neighbors])

    # Draw center point
    ax.plot(center_x, center_y, 'o', color=ARROW_COLOR, markersize=10)
    ax.text(center_x + 0.1, center_y + 0.1, 'Center', fontsize=10,
           color=ARROW_COLOR, ha='left')

    # Draw arrow from boid to center
    ax.annotate('', xy=(center_x, center_y), xytext=(-0.6, -0.4),
               arrowprops=dict(arrowstyle='->', color=ARROW_COLOR,
                              lw=3, mutation_scale=20))

    ax.set_title('Cohesion: Move Toward Group Center',
                fontsize=14, fontweight='bold', color='white', pad=20)

    # Save
    output_path = os.path.join(SCRIPT_DIR, 'cohesion_diagram.png')
    plt.savefig(output_path, bbox_inches='tight', facecolor='#141420',
                edgecolor='none', dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def create_combined_rules_diagram():
    """Create a combined diagram showing all three rules."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)

    rules = [
        ('Separation', 'Avoid Crowding'),
        ('Alignment', 'Match Heading'),
        ('Cohesion', 'Stay Together')
    ]

    for ax, (rule_name, description) in zip(axes, rules):
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')

        # Draw perception circle
        circle = plt.Circle((0, 0), 0.8, fill=False, color=PERCEPTION_COLOR,
                           linestyle='--', linewidth=1.5, alpha=0.5)
        ax.add_patch(circle)

        if rule_name == 'Separation':
            # Central boid
            draw_boid(ax, 0, 0, np.pi/4, BOID_COLOR, 0.15)
            # Close neighbors
            for offset in [(0.25, 0.15), (-0.2, 0.1), (0.05, -0.25)]:
                draw_boid(ax, offset[0], offset[1], np.pi/4, NEIGHBOR_COLOR, 0.1)
                ax.annotate('', xy=(-offset[0]*0.7, -offset[1]*0.7), xytext=(0, 0),
                           arrowprops=dict(arrowstyle='->', color=ARROW_COLOR, lw=1.5))

        elif rule_name == 'Alignment':
            # Central boid
            draw_boid(ax, 0, 0, np.pi, BOID_COLOR, 0.15)
            # Neighbors pointing same way
            avg_angle = np.pi/4
            for offset in [(0.4, 0.2), (-0.3, 0.4), (0.2, -0.3)]:
                draw_boid(ax, offset[0], offset[1], avg_angle, NEIGHBOR_COLOR, 0.1)
            ax.annotate('', xy=(0.5, 0.5), xytext=(0, 0),
                       arrowprops=dict(arrowstyle='->', color=ARROW_COLOR, lw=2))

        elif rule_name == 'Cohesion':
            # Central boid away from group
            draw_boid(ax, -0.5, -0.3, np.pi/4, BOID_COLOR, 0.15)
            # Neighbors clustered
            center = [0.35, 0.35]
            for offset in [(0.2, 0.2), (0.4, 0.3), (0.3, 0.5), (0.5, 0.35)]:
                draw_boid(ax, offset[0], offset[1], np.pi/4, NEIGHBOR_COLOR, 0.1)
            ax.plot(center[0], center[1], 'o', color=ARROW_COLOR, markersize=6)
            ax.annotate('', xy=(center[0], center[1]), xytext=(-0.5, -0.3),
                       arrowprops=dict(arrowstyle='->', color=ARROW_COLOR, lw=2))

        ax.set_title(f'{rule_name}\n{description}',
                    fontsize=12, fontweight='bold', color='white')

    plt.tight_layout()

    # Save
    output_path = os.path.join(SCRIPT_DIR, 'boids_rules_diagram.png')
    plt.savefig(output_path, bbox_inches='tight', facecolor='#141420',
                edgecolor='none', dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def main():
    """Generate all boids conceptual diagrams."""
    print("Generating boids conceptual diagrams...")

    create_separation_diagram()
    create_alignment_diagram()
    create_cohesion_diagram()
    create_combined_rules_diagram()

    print("All diagrams generated successfully!")


if __name__ == '__main__':
    main()
