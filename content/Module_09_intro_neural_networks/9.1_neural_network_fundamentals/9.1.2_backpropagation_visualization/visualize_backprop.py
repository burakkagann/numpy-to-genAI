"""
9.1.2 Visualization Generator

This script generates all the diagrams for the backpropagation exercise:
- xor_problem.png: Shows why XOR cannot be solved with a single line
- network_architecture.png: The 2-4-1 neural network structure
- training_progress.png: Loss curve showing learning over epochs
- forward_pass.png: Data flowing through the network
- backward_pass.png: Gradients flowing backward

Author: NumPy-to-GenAI Project
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.patches as mpatches


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def draw_xor_problem(filename='xor_problem.png'):
    """Show why XOR cannot be separated by a single line."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # XOR data points
    # Class 0: (0,0) and (1,1) - when inputs match
    # Class 1: (0,1) and (1,0) - when inputs differ
    ax.scatter([0, 1], [0, 1], c='#e74c3c', s=300, label='Output = 0',
               edgecolors='black', linewidths=2, zorder=5)
    ax.scatter([0, 1], [1, 0], c='#2ecc71', s=300, label='Output = 1',
               edgecolors='black', linewidths=2, zorder=5)

    # Add labels to points
    ax.annotate('[0,0]', (0, 0), textcoords="offset points",
                xytext=(0, -25), ha='center', fontsize=11)
    ax.annotate('[1,1]', (1, 1), textcoords="offset points",
                xytext=(0, 15), ha='center', fontsize=11)
    ax.annotate('[0,1]', (0, 1), textcoords="offset points",
                xytext=(0, 15), ha='center', fontsize=11)
    ax.annotate('[1,0]', (1, 0), textcoords="offset points",
                xytext=(0, -25), ha='center', fontsize=11)

    # Try to draw a separating line (impossible!)
    x_line = np.linspace(-0.3, 1.3, 100)
    y_line = 0.5 * np.ones_like(x_line)
    ax.plot(x_line, y_line, 'k--', linewidth=2, alpha=0.5, label='Any single line fails')

    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.set_xlabel('Input $x_1$', fontsize=12)
    ax.set_ylabel('Input $x_2$', fontsize=12)
    ax.set_title('The XOR Problem: No Single Line Can Separate These Points',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {filename}")


def draw_network_architecture(filename='network_architecture.png'):
    """Draw a clean neural network diagram."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(-0.5, 4)
    ax.set_ylim(-0.5, 4)
    ax.set_aspect('equal')
    ax.axis('off')

    # Layer positions
    layers = {
        'input': [(0, 1.5), (0, 2.5)],
        'hidden': [(1.5, 0.5), (1.5, 1.5), (1.5, 2.5), (1.5, 3.5)],
        'output': [(3, 2)]
    }

    # Draw connections first (so they appear behind nodes)
    for (x1, y1) in layers['input']:
        for (x2, y2) in layers['hidden']:
            ax.plot([x1, x2], [y1, y2], 'gray', linewidth=1, alpha=0.6)

    for (x1, y1) in layers['hidden']:
        for (x2, y2) in layers['output']:
            ax.plot([x1, x2], [y1, y2], 'gray', linewidth=1, alpha=0.6)

    # Draw nodes
    node_colors = {'input': '#3498db', 'hidden': '#2ecc71', 'output': '#e74c3c'}
    node_labels = {
        'input': ['$x_1$', '$x_2$'],
        'hidden': ['$h_1$', '$h_2$', '$h_3$', '$h_4$'],
        'output': ['$\\hat{y}$']
    }

    for layer_name, positions in layers.items():
        color = node_colors[layer_name]
        labels = node_labels[layer_name]
        for i, (x, y) in enumerate(positions):
            circle = Circle((x, y), 0.25, color=color, ec='black', lw=2, zorder=5)
            ax.add_patch(circle)
            ax.text(x, y, labels[i], ha='center', va='center',
                   fontsize=14, fontweight='bold', color='white', zorder=6)

    # Layer labels
    ax.text(0, -0.2, 'Input Layer\n(2 neurons)', ha='center', fontsize=11,
            fontweight='bold')
    ax.text(1.5, -0.2, 'Hidden Layer\n(4 neurons)', ha='center', fontsize=11,
            fontweight='bold')
    ax.text(3, -0.2, 'Output Layer\n(1 neuron)', ha='center', fontsize=11,
            fontweight='bold')

    # Weight labels
    ax.text(0.75, 2.3, '$W_1$', fontsize=12, ha='center',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.text(2.25, 2.3, '$W_2$', fontsize=12, ha='center',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.set_title('Neural Network Architecture for XOR', fontsize=14,
                 fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {filename}")


def draw_training_progress(filename='training_progress.png'):
    """Train network and plot the loss curve."""
    # XOR data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Initialize
    np.random.seed(42)
    W1 = np.random.randn(2, 4) * 0.5
    W2 = np.random.randn(4, 1) * 0.5

    losses = []
    epochs = 1000

    for epoch in range(epochs):
        # Forward
        a1 = sigmoid(X @ W1)
        a2 = sigmoid(a1 @ W2)
        loss = np.mean((a2 - y) ** 2)
        losses.append(loss)

        # Backward
        dz2 = a2 - y
        dW2 = a1.T @ dz2 / 4
        dz1 = (dz2 @ W2.T) * (a1 * (1 - a1))
        dW1 = X.T @ dz1 / 4

        # Update
        W2 -= 1.0 * dW2
        W1 -= 1.0 * dW1

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(losses, linewidth=2, color='#3498db')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (Mean Squared Error)', fontsize=12)
    ax.set_title('Training Progress: Loss Decreases as Network Learns XOR',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Annotations
    ax.annotate('High loss at start\n(random predictions)',
               xy=(0, losses[0]), xytext=(100, losses[0] + 0.02),
               arrowprops=dict(arrowstyle='->', color='gray'),
               fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.annotate('Network has learned XOR!',
               xy=(900, losses[-1]), xytext=(600, losses[-1] + 0.05),
               arrowprops=dict(arrowstyle='->', color='gray'),
               fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {filename}")


def draw_forward_pass(filename='forward_pass.png'):
    """Visualize data flowing forward through the network."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(-0.5, 5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Train network first to get meaningful values
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    np.random.seed(42)
    W1 = np.random.randn(2, 4) * 0.5
    W2 = np.random.randn(4, 1) * 0.5

    for _ in range(1000):
        a1 = sigmoid(X @ W1)
        a2 = sigmoid(a1 @ W2)
        dz2 = a2 - y
        dW2 = a1.T @ dz2 / 4
        dz1 = (dz2 @ W2.T) * (a1 * (1 - a1))
        dW1 = X.T @ dz1 / 4
        W2 -= 1.0 * dW2
        W1 -= 1.0 * dW1

    # Use input [0, 1] which should predict 1
    x_sample = np.array([[0, 1]])
    a1_sample = sigmoid(x_sample @ W1)
    a2_sample = sigmoid(a1_sample @ W2)

    # Layer positions
    layers = {
        'input': [(0.5, 1.5), (0.5, 3)],
        'hidden': [(2, 0.5), (2, 1.5), (2, 2.5), (2, 3.5)],
        'output': [(3.5, 2)]
    }

    # Draw arrows showing data flow
    for i, (x1, y1) in enumerate(layers['input']):
        for j, (x2, y2) in enumerate(layers['hidden']):
            ax.annotate('', xy=(x2 - 0.3, y2), xytext=(x1 + 0.3, y1),
                       arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.5, alpha=0.6))

    for i, (x1, y1) in enumerate(layers['hidden']):
        for j, (x2, y2) in enumerate(layers['output']):
            ax.annotate('', xy=(x2 - 0.3, y2), xytext=(x1 + 0.3, y1),
                       arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.5, alpha=0.6))

    # Draw nodes with values
    # Input layer
    input_vals = [0, 1]
    for i, (px, py) in enumerate(layers['input']):
        circle = Circle((px, py), 0.28, color='#3498db', ec='black', lw=2, zorder=5)
        ax.add_patch(circle)
        ax.text(px, py, f'{input_vals[i]}', ha='center', va='center',
               fontsize=14, fontweight='bold', color='white', zorder=6)
        ax.text(px, py + 0.5, f'$x_{i+1}$', ha='center', fontsize=11)

    # Hidden layer
    for i, (px, py) in enumerate(layers['hidden']):
        val = a1_sample[0, i]
        circle = Circle((px, py), 0.28, color='#2ecc71', ec='black', lw=2, zorder=5)
        ax.add_patch(circle)
        ax.text(px, py, f'{val:.2f}', ha='center', va='center',
               fontsize=11, fontweight='bold', color='white', zorder=6)

    # Output layer
    px, py = layers['output'][0]
    circle = Circle((px, py), 0.28, color='#e74c3c', ec='black', lw=2, zorder=5)
    ax.add_patch(circle)
    ax.text(px, py, f'{a2_sample[0,0]:.2f}', ha='center', va='center',
           fontsize=14, fontweight='bold', color='white', zorder=6)
    ax.text(px, py + 0.5, '$\\hat{y}$', ha='center', fontsize=11)

    # Flow direction indicator
    ax.annotate('Data flows forward', xy=(2, -0.2), fontsize=12,
               ha='center', color='#27ae60', fontweight='bold')
    ax.annotate('', xy=(3.2, -0.35), xytext=(0.8, -0.35),
               arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2))

    ax.set_title(f'Forward Pass: Input [0, 1] produces prediction {a2_sample[0,0]:.2f} (target: 1)',
                fontsize=13, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {filename}")


def draw_backward_pass(filename='backward_pass.png'):
    """Visualize gradients flowing backward through the network."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(-0.5, 5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Layer positions
    layers = {
        'input': [(0.5, 1.5), (0.5, 3)],
        'hidden': [(2, 0.5), (2, 1.5), (2, 2.5), (2, 3.5)],
        'output': [(3.5, 2)]
    }

    # Draw backward arrows (right to left)
    for i, (x1, y1) in enumerate(layers['hidden']):
        for j, (x2, y2) in enumerate(layers['output']):
            ax.annotate('', xy=(x1 + 0.3, y1), xytext=(x2 - 0.3, y2),
                       arrowprops=dict(arrowstyle='->', color='#c0392b', lw=2, alpha=0.7))

    for i, (x1, y1) in enumerate(layers['input']):
        for j, (x2, y2) in enumerate(layers['hidden']):
            ax.annotate('', xy=(x1 + 0.3, y1), xytext=(x2 - 0.3, y2),
                       arrowprops=dict(arrowstyle='->', color='#c0392b', lw=2, alpha=0.7))

    # Draw nodes
    for px, py in layers['input']:
        circle = Circle((px, py), 0.28, color='#3498db', ec='black', lw=2, zorder=5)
        ax.add_patch(circle)

    for px, py in layers['hidden']:
        circle = Circle((px, py), 0.28, color='#e67e22', ec='black', lw=2, zorder=5)
        ax.add_patch(circle)
        ax.text(px, py, '$\\delta$', ha='center', va='center',
               fontsize=11, fontweight='bold', color='white', zorder=6)

    px, py = layers['output'][0]
    circle = Circle((px, py), 0.28, color='#c0392b', ec='black', lw=2, zorder=5)
    ax.add_patch(circle)
    ax.text(px, py, 'err', ha='center', va='center',
           fontsize=10, fontweight='bold', color='white', zorder=6)

    # Annotations
    ax.text(4.2, 2, 'Error = prediction - target', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

    ax.text(1.2, 4.2, 'Chain Rule:', fontsize=11, fontweight='bold')
    ax.text(1.2, 3.8, 'Each hidden neuron gets\na share of the blame', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Flow direction indicator
    ax.annotate('Gradients flow backward', xy=(2, -0.2), fontsize=12,
               ha='center', color='#c0392b', fontweight='bold')
    ax.annotate('', xy=(0.8, -0.35), xytext=(3.2, -0.35),
               arrowprops=dict(arrowstyle='->', color='#c0392b', lw=2))

    ax.set_title('Backward Pass: Error Propagates Back Through the Network',
                fontsize=13, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {filename}")


if __name__ == '__main__':
    print("Generating visualizations for 9.1.2 Backpropagation Visualization...")
    print()

    draw_xor_problem('xor_problem.png')
    draw_network_architecture('network_architecture.png')
    draw_training_progress('training_progress.png')
    draw_forward_pass('forward_pass.png')
    draw_backward_pass('backward_pass.png')

    print()
    print("All visualizations generated successfully!")
