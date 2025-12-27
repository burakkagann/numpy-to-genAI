import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# =============================================================================
# Activation Functions
# =============================================================================

def sigmoid(x):
    """Sigmoid: squashes input to (0, 1)"""
    return 1.0 / (1.0 + np.exp(-x))

def relu(x):
    """ReLU: max(0, x)"""
    return np.maximum(0, x)

def tanh_activation(x):
    """Tanh: squashes input to (-1, 1)"""
    return np.tanh(x)

# =============================================================================
# Generate Individual Patterns
# =============================================================================

def create_pattern(activation_func, name, size=400):
    """Create a single activation function pattern."""
    x = np.linspace(-5, 5, size)
    y = np.linspace(-5, 5, size)
    X, Y = np.meshgrid(x, y)

    # Use distance from center as input
    distance = np.sqrt(X**2 + Y**2)

    # Apply activation function
    values = activation_func(distance - 2.5)

    # Normalize to 0-255 range
    if name == 'relu':
        # ReLU needs special handling since it's unbounded
        values = np.clip(values / 3.0, 0, 1)
    elif name == 'tanh':
        # Tanh outputs are in [-1, 1], shift to [0, 1]
        values = (values + 1) / 2
    # Sigmoid is already in [0, 1]

    values = (values * 255).astype(np.uint8)

    # Create grayscale image
    img = Image.fromarray(values, mode='L')
    return img

# Generate individual patterns
print("Generating individual patterns...")

sigmoid_img = create_pattern(sigmoid, 'sigmoid')
sigmoid_img.save('sigmoid_pattern.png')
print("  Saved sigmoid_pattern.png")

relu_img = create_pattern(relu, 'relu')
relu_img.save('relu_pattern.png')
print("  Saved relu_pattern.png")

tanh_img = create_pattern(tanh_activation, 'tanh')
tanh_img.save('tanh_pattern.png')
print("  Saved tanh_pattern.png")

# =============================================================================
# Create Comparison Grid
# =============================================================================

print("Creating comparison grid...")

# Create figure with 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=100)
fig.patch.set_facecolor('black')

# Load and display each pattern
patterns = [
    ('Sigmoid', np.array(sigmoid_img)),
    ('ReLU', np.array(relu_img)),
    ('Tanh', np.array(tanh_img)),
    ('Combined (RGB)', np.array(Image.open('activation_art.png')))
]

for idx, (title, img) in enumerate(patterns):
    ax = axes[idx // 2, idx % 2]
    if len(img.shape) == 2:
        ax.imshow(img, cmap='viridis')
    else:
        ax.imshow(img)
    ax.set_title(title, fontsize=14, fontweight='bold', color='white', pad=10)
    ax.axis('off')

plt.tight_layout(pad=2)
plt.savefig('comparison_grid.png', dpi=100, facecolor='black',
            bbox_inches='tight', pad_inches=0.5)
plt.close()
print("  Saved comparison_grid.png")

# =============================================================================
# Create Activation Curves Diagram
# =============================================================================

print("Creating activation curves diagram...")

fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

x = np.linspace(-5, 5, 200)

# Plot each activation function
ax.plot(x, sigmoid(x), 'b-', linewidth=2.5, label='Sigmoid: 1/(1+e^(-x))')
ax.plot(x, relu(x), 'r-', linewidth=2.5, label='ReLU: max(0, x)')
ax.plot(x, tanh_activation(x), 'g-', linewidth=2.5, label='Tanh: tanh(x)')

# Add reference lines
ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
ax.axhline(y=1, color='gray', linestyle=':', linewidth=0.5, alpha=0.7)
ax.axhline(y=-1, color='gray', linestyle=':', linewidth=0.5, alpha=0.7)
ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

# Labels and title
ax.set_xlabel('Input (x)', fontsize=12)
ax.set_ylabel('Output f(x)', fontsize=12)
ax.set_title('Common Activation Functions in Neural Networks',
             fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(-5, 5)
ax.set_ylim(-1.5, 3)

plt.tight_layout()
plt.savefig('activation_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved activation_curves.png")

print("\nAll visualizations generated successfully!")
