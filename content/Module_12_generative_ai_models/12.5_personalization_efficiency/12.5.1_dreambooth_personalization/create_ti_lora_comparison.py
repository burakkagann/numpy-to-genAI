"""
Create TI vs LoRA Comparison Grid

Generates a side-by-side comparison of Textual Inversion vs LoRA training progress.
"""

from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Paths
MODULE_DIR = Path(__file__).parent
TI_DIR = MODULE_DIR / "training_progress_ti"
LORA_DIR = MODULE_DIR / "training_progress_lora"
OUTPUTS_DIR = MODULE_DIR / "outputs"


def create_ti_lora_comparison():
    """Create comparison of TI vs LoRA outputs."""

    # Get TI images
    ti_images = sorted(list(TI_DIR.glob('*.png')))
    lora_images = sorted(list(LORA_DIR.glob('*.png')))

    if not ti_images or not lora_images:
        print("Missing training progress images!")
        print(f"  TI images: {len(ti_images)}")
        print(f"  LoRA images: {len(lora_images)}")
        return None

    # Take last image from each (best quality)
    ti_final = ti_images[-1]
    lora_final = lora_images[-1]

    print(f"TI final: {ti_final.name}")
    print(f"LoRA final: {lora_final.name}")

    # Create comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=150)

    # TI output
    ti_img = Image.open(ti_final)
    axes[0].imshow(ti_img)
    axes[0].set_title(f"Textual Inversion\n({ti_final.stem})", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # LoRA output
    lora_img = Image.open(lora_final)
    axes[1].imshow(lora_img)
    axes[1].set_title(f"LoRA\n({lora_final.stem})", fontsize=12, fontweight='bold')
    axes[1].axis('off')

    fig.suptitle("Training Method Comparison: Textual Inversion vs LoRA",
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Save
    OUTPUTS_DIR.mkdir(exist_ok=True)
    output_path = OUTPUTS_DIR / "exercise3_ti_vs_lora.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\nSaved: {output_path}")
    return output_path


def create_training_progression():
    """Create training progression comparison."""

    ti_images = sorted(list(TI_DIR.glob('*.png')))
    lora_images = sorted(list(LORA_DIR.glob('*.png')))

    # Get up to 5 images from each
    ti_samples = ti_images[:5] if len(ti_images) >= 5 else ti_images
    lora_samples = lora_images[:5] if len(lora_images) >= 5 else lora_images

    max_cols = max(len(ti_samples), len(lora_samples))

    if max_cols == 0:
        print("No training progress images found!")
        return None

    fig, axes = plt.subplots(2, max_cols, figsize=(3*max_cols, 7), dpi=150)

    # TI row
    for idx, img_path in enumerate(ti_samples):
        img = Image.open(img_path)
        axes[0, idx].imshow(img)
        axes[0, idx].set_title(img_path.stem.replace('step_', 'Step '), fontsize=9)
        axes[0, idx].axis('off')

    # Hide unused TI cells
    for idx in range(len(ti_samples), max_cols):
        axes[0, idx].axis('off')

    # LoRA row
    for idx, img_path in enumerate(lora_samples):
        img = Image.open(img_path)
        axes[1, idx].imshow(img)
        axes[1, idx].set_title(img_path.stem.replace('step_', 'Step '), fontsize=9)
        axes[1, idx].axis('off')

    # Hide unused LoRA cells
    for idx in range(len(lora_samples), max_cols):
        axes[1, idx].axis('off')

    # Row labels
    axes[0, 0].set_ylabel("Textual\nInversion", fontsize=11, fontweight='bold', rotation=0, labelpad=50, va='center')
    axes[1, 0].set_ylabel("LoRA", fontsize=11, fontweight='bold', rotation=0, labelpad=50, va='center')

    fig.suptitle("Training Progression: Textual Inversion vs LoRA",
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9, left=0.1)

    # Save
    output_path = OUTPUTS_DIR / "training_progression_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved: {output_path}")
    return output_path


if __name__ == "__main__":
    print("=" * 60)
    print("Creating TI vs LoRA Comparisons")
    print("=" * 60)

    print("\n1. Creating final output comparison...")
    create_ti_lora_comparison()

    print("\n2. Creating training progression comparison...")
    create_training_progression()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
