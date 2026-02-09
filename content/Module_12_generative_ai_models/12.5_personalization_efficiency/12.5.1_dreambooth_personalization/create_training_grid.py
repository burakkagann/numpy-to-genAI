"""
Create Training Samples Grid

Generates a 3x3 grid of training images for documentation display.
"""

import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Paths
MODULE_DIR = Path(__file__).parent
TRAINING_DIR = MODULE_DIR / "training_images"
OUTPUTS_DIR = MODULE_DIR / "outputs"

def create_training_grid():
    """Create a 3x3 grid of training samples."""

    # Find training images (support both jpg and png)
    training_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        training_images.extend(list(TRAINING_DIR.glob(ext)))

    # Sort by name
    training_images = sorted([f for f in training_images if f.name != 'README.md'])

    if len(training_images) < 9:
        print(f"Warning: Only found {len(training_images)} training images, need 9 for grid")
        # Pad with duplicates if needed
        while len(training_images) < 9:
            training_images.append(training_images[len(training_images) % len(training_images)])

    # Take first 9
    training_images = training_images[:9]

    print(f"Creating grid from {len(training_images)} images:")
    for img in training_images:
        print(f"  - {img.name}")

    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(10, 10), dpi=150)
    fig.suptitle("Training Dataset: African Fabric Patterns (9 of 10)",
                 fontsize=14, fontweight='bold', y=0.98)

    for idx, (ax, img_path) in enumerate(zip(axes.flatten(), training_images)):
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"Sample {idx+1}", fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    # Save
    OUTPUTS_DIR.mkdir(exist_ok=True)
    output_path = OUTPUTS_DIR / "training_samples_grid.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\nSaved: {output_path}")
    return output_path


def create_comparison_grid():
    """Create side-by-side comparison: training vs generated."""

    # Find training images
    training_images = sorted(list(TRAINING_DIR.glob('*.jpg')) + list(TRAINING_DIR.glob('*.png')))
    training_images = [f for f in training_images if f.name != 'README.md'][:9]

    # Find generated images from exercise1
    generated_dir = OUTPUTS_DIR / "exercise1_variations"
    if generated_dir.exists():
        generated_images = sorted(list(generated_dir.glob('*.png')))[:9]
    else:
        print("No generated images found. Run exercise1_generate.py first.")
        return None

    if len(generated_images) < 9:
        print(f"Only found {len(generated_images)} generated images")
        return None

    # Create comparison figure
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), dpi=150)

    # Training images row (as a grid within subplot)
    ax_train = axes[0]
    ax_train.axis('off')
    ax_train.set_title("Training Data (9 African Fabric Samples)", fontsize=12, fontweight='bold', pad=10)

    # Create inner grid for training
    inner_grid_train = fig.add_gridspec(3, 3, left=0.02, right=0.98, top=0.88, bottom=0.52,
                                         wspace=0.02, hspace=0.02)
    for idx, img_path in enumerate(training_images):
        ax = fig.add_subplot(inner_grid_train[idx // 3, idx % 3])
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis('off')

    # Generated images row
    ax_gen = axes[1]
    ax_gen.axis('off')
    ax_gen.set_title("Generated Output (DreamBooth Personalized)", fontsize=12, fontweight='bold', pad=10)

    # Create inner grid for generated
    inner_grid_gen = fig.add_gridspec(3, 3, left=0.02, right=0.98, top=0.46, bottom=0.02,
                                       wspace=0.02, hspace=0.02)
    for idx, img_path in enumerate(generated_images):
        ax = fig.add_subplot(inner_grid_gen[idx // 3, idx % 3])
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis('off')

    # Save
    output_path = OUTPUTS_DIR / "training_vs_generated.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved: {output_path}")
    return output_path


if __name__ == "__main__":
    print("=" * 60)
    print("Creating Training & Comparison Grids")
    print("=" * 60)

    # Create training samples grid
    print("\n1. Creating training samples grid...")
    create_training_grid()

    # Create comparison grid
    print("\n2. Creating training vs generated comparison...")
    create_comparison_grid()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
