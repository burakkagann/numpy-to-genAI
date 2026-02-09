"""
Generate Conceptual Diagrams for Module 12.5.1

This script creates educational diagrams explaining:
1. DreamBooth architecture and workflow
2. Prior preservation concept
3. Comparison of Textual Inversion vs LoRA
4. Knowledge transfer from DDPM Basics (Module 12.3.1)

These diagrams are for documentation purposes and help learners
visualize the key concepts in DreamBooth personalization.
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patches import Rectangle, Circle, Arrow
import matplotlib.lines as mlines

# Output directory
OUTPUT_DIR = Path(__file__).parent / "conceptual_diagrams"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_dreambooth_architecture_diagram():
    """
    Create a diagram showing the DreamBooth training workflow.
    Shows how subject images + prompts are used to fine-tune the U-Net.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(7, 9.5, "DreamBooth Training Architecture", fontsize=16,
            fontweight='bold', ha='center', va='center')

    # Color scheme
    colors = {
        'input': '#e3f2fd',      # Light blue
        'model': '#fff3e0',       # Light orange
        'process': '#e8f5e9',     # Light green
        'output': '#fce4ec',      # Light pink
        'frozen': '#f5f5f5',      # Light gray
        'trainable': '#c8e6c9'    # Green
    }

    # Input: Subject Images
    rect1 = FancyBboxPatch((0.5, 6), 2.5, 2.5, boxstyle="round,pad=0.1",
                            facecolor=colors['input'], edgecolor='#1976d2', linewidth=2)
    ax.add_patch(rect1)
    ax.text(1.75, 7.75, "Subject Images", fontsize=10, fontweight='bold', ha='center')
    ax.text(1.75, 7.25, "(3-10 photos of\nyour subject)", fontsize=8, ha='center')

    # Small images representation
    for i in range(3):
        rect_small = Rectangle((0.8 + i*0.6, 6.3), 0.5, 0.5,
                                facecolor='white', edgecolor='gray', linewidth=1)
        ax.add_patch(rect_small)
        ax.text(1.05 + i*0.6, 6.55, f"img{i+1}", fontsize=6, ha='center')

    # Input: Instance Prompt
    rect2 = FancyBboxPatch((0.5, 3.5), 2.5, 2, boxstyle="round,pad=0.1",
                            facecolor=colors['input'], edgecolor='#1976d2', linewidth=2)
    ax.add_patch(rect2)
    ax.text(1.75, 5, "Instance Prompt", fontsize=10, fontweight='bold', ha='center')
    ax.text(1.75, 4.3, '"a sks african\nfabric pattern"', fontsize=8, ha='center', style='italic')

    # Arrow from inputs to text encoder
    ax.annotate('', xy=(3.5, 4.5), xytext=(3, 4.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # Text Encoder (frozen)
    rect3 = FancyBboxPatch((3.5, 3.5), 2, 2, boxstyle="round,pad=0.1",
                            facecolor=colors['frozen'], edgecolor='#9e9e9e', linewidth=2)
    ax.add_patch(rect3)
    ax.text(4.5, 5, "Text Encoder", fontsize=10, fontweight='bold', ha='center')
    ax.text(4.5, 4.3, "(CLIP)\nFROZEN", fontsize=8, ha='center', color='gray')

    # VAE Encoder (frozen)
    rect4 = FancyBboxPatch((3.5, 6), 2, 2.5, boxstyle="round,pad=0.1",
                            facecolor=colors['frozen'], edgecolor='#9e9e9e', linewidth=2)
    ax.add_patch(rect4)
    ax.text(4.5, 7.75, "VAE Encoder", fontsize=10, fontweight='bold', ha='center')
    ax.text(4.5, 7, "Image → Latent\nFROZEN", fontsize=8, ha='center', color='gray')

    # Arrow from image to VAE
    ax.annotate('', xy=(3.5, 7.25), xytext=(3, 7.25),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # Arrows to U-Net
    ax.annotate('', xy=(6, 5.5), xytext=(5.5, 4.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.annotate('', xy=(6, 6.5), xytext=(5.5, 7.25),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # U-Net (TRAINABLE)
    rect5 = FancyBboxPatch((6, 4.5), 3, 3.5, boxstyle="round,pad=0.1",
                            facecolor=colors['trainable'], edgecolor='#388e3c', linewidth=3)
    ax.add_patch(rect5)
    ax.text(7.5, 7.5, "U-Net", fontsize=14, fontweight='bold', ha='center')
    ax.text(7.5, 6.7, "Noise Prediction\nNetwork", fontsize=10, ha='center')
    ax.text(7.5, 5.8, "TRAINABLE", fontsize=10, fontweight='bold',
            ha='center', color='#2e7d32')
    ax.text(7.5, 5.2, "(with LoRA adapters\nor full fine-tune)", fontsize=8, ha='center')

    # Noise input
    ax.text(7.5, 3.8, "🎲 Noise", fontsize=10, ha='center')
    ax.annotate('', xy=(7.5, 4.5), xytext=(7.5, 4),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # Arrow to loss
    ax.annotate('', xy=(9.5, 6), xytext=(9, 6),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # Loss calculation
    rect6 = FancyBboxPatch((9.5, 4.5), 2.5, 3.5, boxstyle="round,pad=0.1",
                            facecolor=colors['process'], edgecolor='#43a047', linewidth=2)
    ax.add_patch(rect6)
    ax.text(10.75, 7.5, "Training Loss", fontsize=10, fontweight='bold', ha='center')
    ax.text(10.75, 6.7, "MSE(ε_pred, ε_true)", fontsize=9, ha='center', family='monospace')
    ax.text(10.75, 5.8, "+", fontsize=14, ha='center')
    ax.text(10.75, 5.2, "Prior Preservation", fontsize=9, ha='center')
    ax.text(10.75, 4.8, "(prevents forgetting)", fontsize=8, ha='center', style='italic')

    # Arrow back to U-Net (gradient)
    ax.annotate('', xy=(9, 5), xytext=(9.5, 5),
                arrowprops=dict(arrowstyle='->', color='#d32f2f', lw=2))
    ax.text(9.25, 4.5, "Backprop", fontsize=8, ha='center', color='#d32f2f')

    # Output box
    rect7 = FancyBboxPatch((10, 1.5), 3, 2, boxstyle="round,pad=0.1",
                            facecolor=colors['output'], edgecolor='#c2185b', linewidth=2)
    ax.add_patch(rect7)
    ax.text(11.5, 3, "Personalized Model", fontsize=10, fontweight='bold', ha='center')
    ax.text(11.5, 2.3, '"sks" = your subject', fontsize=9, ha='center', style='italic')
    ax.text(11.5, 1.8, "LoRA: ~10-50 MB\nFull: ~5 GB", fontsize=8, ha='center')

    # Arrow to output
    ax.annotate('', xy=(10.75, 3.5), xytext=(10.75, 4.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=colors['frozen'], edgecolor='#9e9e9e',
                       label='Frozen (not trained)'),
        mpatches.Patch(facecolor=colors['trainable'], edgecolor='#388e3c',
                       label='Trainable (updated)'),
        mpatches.Patch(facecolor=colors['input'], edgecolor='#1976d2',
                       label='Input'),
        mpatches.Patch(facecolor=colors['process'], edgecolor='#43a047',
                       label='Training Process'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=8)

    # Note about connection to DDPM
    ax.text(7, 0.8, "Connection to DDPM (Module 12.3.1):", fontsize=9,
            fontweight='bold', ha='center')
    ax.text(7, 0.4, "Same U-Net architecture, same noise prediction loss (MSE) — "
            "but now conditioned on text + fine-tuned on your subject",
            fontsize=8, ha='center', style='italic')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dreambooth_architecture.png", dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'dreambooth_architecture.png'}")


def create_prior_preservation_diagram():
    """
    Create a diagram explaining prior preservation loss.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Color scheme
    colors = {
        'good': '#c8e6c9',
        'bad': '#ffcdd2',
        'neutral': '#e0e0e0'
    }

    # Left panel: Without Prior Preservation
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(5, 9.5, "WITHOUT Prior Preservation", fontsize=14,
            fontweight='bold', ha='center', color='#d32f2f')

    # Training data
    rect1 = FancyBboxPatch((0.5, 6), 3, 2.5, boxstyle="round,pad=0.1",
                            facecolor=colors['neutral'], edgecolor='gray', linewidth=2)
    ax.add_patch(rect1)
    ax.text(2, 7.75, "Training Only On", fontsize=10, fontweight='bold', ha='center')
    ax.text(2, 7, "YOUR Subject", fontsize=10, ha='center')
    ax.text(2, 6.4, "(10 images)", fontsize=9, ha='center', style='italic')

    # Arrow
    ax.annotate('', xy=(5, 7.25), xytext=(3.5, 7.25),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # Model
    rect2 = FancyBboxPatch((5, 6), 3, 2.5, boxstyle="round,pad=0.1",
                            facecolor=colors['bad'], edgecolor='#d32f2f', linewidth=2)
    ax.add_patch(rect2)
    ax.text(6.5, 7.75, "Fine-tuned Model", fontsize=10, fontweight='bold', ha='center')
    ax.text(6.5, 7, "OVERFITS!", fontsize=10, ha='center', color='#d32f2f')

    # Results
    ax.text(5, 5, "Results:", fontsize=11, fontweight='bold')

    ax.text(1, 4, "✓ Knows your subject", fontsize=10, color='#388e3c')
    ax.text(1, 3.2, "✗ Forgets other dogs/fabrics", fontsize=10, color='#d32f2f')
    ax.text(1, 2.4, "✗ Language drift", fontsize=10, color='#d32f2f')
    ax.text(1, 1.6, '✗ "dog" → only YOUR dog', fontsize=10, color='#d32f2f')

    ax.text(5, 0.8, "Problem: Model loses general knowledge",
            fontsize=10, ha='center', style='italic', color='#d32f2f')

    # Right panel: With Prior Preservation
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(5, 9.5, "WITH Prior Preservation", fontsize=14,
            fontweight='bold', ha='center', color='#388e3c')

    # Subject data
    rect3 = FancyBboxPatch((0.5, 7), 3, 1.8, boxstyle="round,pad=0.1",
                            facecolor='#e3f2fd', edgecolor='#1976d2', linewidth=2)
    ax.add_patch(rect3)
    ax.text(2, 8.3, "YOUR Subject", fontsize=10, fontweight='bold', ha='center')
    ax.text(2, 7.6, "(10 images)", fontsize=9, ha='center', style='italic')
    ax.text(2, 7.15, '"sks fabric"', fontsize=9, ha='center')

    # Class data
    rect4 = FancyBboxPatch((0.5, 5), 3, 1.8, boxstyle="round,pad=0.1",
                            facecolor='#fff3e0', edgecolor='#ff9800', linewidth=2)
    ax.add_patch(rect4)
    ax.text(2, 6.3, "Class Images", fontsize=10, fontweight='bold', ha='center')
    ax.text(2, 5.6, "(100 generated)", fontsize=9, ha='center', style='italic')
    ax.text(2, 5.15, '"fabric pattern"', fontsize=9, ha='center')

    # Arrows
    ax.annotate('', xy=(4, 7.5), xytext=(3.5, 7.9),
                arrowprops=dict(arrowstyle='->', color='#1976d2', lw=2))
    ax.annotate('', xy=(4, 6.5), xytext=(3.5, 5.9),
                arrowprops=dict(arrowstyle='->', color='#ff9800', lw=2))

    # Loss box
    rect5 = FancyBboxPatch((4, 5.5), 2.5, 3, boxstyle="round,pad=0.1",
                            facecolor=colors['neutral'], edgecolor='gray', linewidth=2)
    ax.add_patch(rect5)
    ax.text(5.25, 8, "Combined Loss", fontsize=10, fontweight='bold', ha='center')
    ax.text(5.25, 7.2, "L_subject", fontsize=10, ha='center', color='#1976d2')
    ax.text(5.25, 6.7, "+", fontsize=12, ha='center')
    ax.text(5.25, 6.2, "λ × L_prior", fontsize=10, ha='center', color='#ff9800')

    # Arrow to model
    ax.annotate('', xy=(7, 7), xytext=(6.5, 7),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # Model
    rect6 = FancyBboxPatch((7, 5.5), 2.5, 3, boxstyle="round,pad=0.1",
                            facecolor=colors['good'], edgecolor='#388e3c', linewidth=2)
    ax.add_patch(rect6)
    ax.text(8.25, 8, "Balanced Model", fontsize=10, fontweight='bold', ha='center')
    ax.text(8.25, 7.2, "Learns subject", fontsize=9, ha='center')
    ax.text(8.25, 6.7, "+", fontsize=10, ha='center')
    ax.text(8.25, 6.2, "Keeps general", fontsize=9, ha='center')
    ax.text(8.25, 5.7, "knowledge", fontsize=9, ha='center')

    # Results
    ax.text(5, 4.2, "Results:", fontsize=11, fontweight='bold')

    ax.text(1, 3.4, "✓ Knows your subject", fontsize=10, color='#388e3c')
    ax.text(1, 2.6, "✓ Still knows other fabrics", fontsize=10, color='#388e3c')
    ax.text(1, 1.8, "✓ No language drift", fontsize=10, color='#388e3c')
    ax.text(1, 1.0, "✓ Generalizes to new contexts", fontsize=10, color='#388e3c')

    ax.text(5, 0.3, "Solution: Train on BOTH subject AND generic class",
            fontsize=10, ha='center', style='italic', color='#388e3c')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "prior_preservation_demo.png", dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'prior_preservation_demo.png'}")


def create_ti_vs_lora_comparison():
    """
    Create a comparison diagram of Textual Inversion vs LoRA.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(7, 9.5, "Textual Inversion vs LoRA vs Full Fine-tune",
            fontsize=16, fontweight='bold', ha='center')

    # Colors
    colors = {
        'ti': '#e3f2fd',
        'lora': '#e8f5e9',
        'full': '#fff3e0'
    }

    # Column positions
    col_x = [2, 7, 12]
    methods = ["Textual Inversion", "LoRA (Recommended)", "Full Fine-tune"]
    method_colors = [colors['ti'], colors['lora'], colors['full']]
    edge_colors = ['#1976d2', '#388e3c', '#ff9800']

    for i, (x, method, bg_color, edge_color) in enumerate(zip(col_x, methods, method_colors, edge_colors)):
        # Header box
        rect = FancyBboxPatch((x-1.8, 7.5), 3.6, 1.5, boxstyle="round,pad=0.1",
                               facecolor=bg_color, edgecolor=edge_color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, 8.5, method, fontsize=11, fontweight='bold', ha='center')

        # Highlight recommended
        if "Recommended" in method:
            ax.text(x, 7.7, "★ Best Balance ★", fontsize=9, ha='center',
                    color='#388e3c', fontweight='bold')

    # Comparison rows
    row_labels = [
        "What trains:",
        "Parameters:",
        "Output size:",
        "Training time:",
        "Quality:",
        "Flexibility:",
        "Sharing:"
    ]

    ti_values = [
        "Token embedding only",
        "~768",
        "~3 KB",
        "30-60 min",
        "Good for styles",
        "Limited",
        "Very easy (tiny file)"
    ]

    lora_values = [
        "U-Net attention adapters",
        "~4-40 million",
        "10-50 MB",
        "15-30 min",
        "Very good",
        "High (combinable)",
        "Easy (small file)"
    ]

    full_values = [
        "Entire U-Net + Text Enc.",
        "~1 billion",
        "~5 GB",
        "1-2 hours",
        "Excellent",
        "Highest",
        "Hard (huge file)"
    ]

    all_values = [ti_values, lora_values, full_values]

    y_start = 6.8
    y_step = 0.8

    for row_idx, label in enumerate(row_labels):
        y = y_start - row_idx * y_step

        # Row label
        ax.text(0.2, y, label, fontsize=10, fontweight='bold', va='center')

        # Values
        for col_idx, (x, values) in enumerate(zip(col_x, all_values)):
            value = values[row_idx]

            # Color code quality
            if "Good" in value or "Very" in value or "Excellent" in value:
                color = '#388e3c'
            elif "Limited" in value or "Hard" in value:
                color = '#d32f2f'
            else:
                color = 'black'

            ax.text(x, y, value, fontsize=9, ha='center', va='center', color=color)

    # Visual representation of what gets trained
    ax.text(7, 0.8, "Visual: What gets trained", fontsize=11, fontweight='bold', ha='center')

    # TI: single embedding
    ax.add_patch(Rectangle((0.5, 0.2), 0.3, 0.3, facecolor='#1976d2'))
    ax.text(2, 0.35, "= 1 embedding vector", fontsize=8, va='center')

    # LoRA: attention matrices
    for j in range(4):
        ax.add_patch(Rectangle((5.5 + j*0.4, 0.2), 0.3, 0.3, facecolor='#388e3c'))
    ax.text(8, 0.35, "= attention adapters", fontsize=8, va='center')

    # Full: everything
    for j in range(12):
        ax.add_patch(Rectangle((10 + j*0.25, 0.2), 0.2, 0.3, facecolor='#ff9800'))
    ax.text(13.5, 0.35, "= entire model", fontsize=8, va='center')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ti_vs_lora_comparison.png", dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'ti_vs_lora_comparison.png'}")


def create_knowledge_transfer_diagram():
    """
    Create a diagram showing the connection to DDPM Basics (Module 12.3.1).
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(7, 9.5, "Knowledge Transfer: From DDPM (12.3.1) to DreamBooth (12.5.1)",
            fontsize=14, fontweight='bold', ha='center')

    # DDPM box (left)
    rect1 = FancyBboxPatch((0.5, 3), 5.5, 5.5, boxstyle="round,pad=0.1",
                            facecolor='#e3f2fd', edgecolor='#1976d2', linewidth=2)
    ax.add_patch(rect1)
    ax.text(3.25, 8, "Module 12.3.1: DDPM Basics", fontsize=12,
            fontweight='bold', ha='center')

    ddpm_items = [
        "• Forward diffusion: add noise",
        "• Reverse diffusion: predict noise",
        "• U-Net architecture",
        "• MSE loss: ||ε - ε_θ||²",
        "• Unconditional generation",
        "• Random samples from learned distribution"
    ]

    for i, item in enumerate(ddpm_items):
        ax.text(1, 7 - i*0.6, item, fontsize=9, va='center')

    # DreamBooth box (right)
    rect2 = FancyBboxPatch((8, 3), 5.5, 5.5, boxstyle="round,pad=0.1",
                            facecolor='#e8f5e9', edgecolor='#388e3c', linewidth=2)
    ax.add_patch(rect2)
    ax.text(10.75, 8, "Module 12.5.1: DreamBooth", fontsize=12,
            fontweight='bold', ha='center')

    dreambooth_items = [
        "• Same forward diffusion ✓",
        "• Same reverse diffusion ✓",
        "• Same U-Net + LoRA adapters",
        "• Same MSE loss + prior loss",
        "• Text-CONDITIONAL generation",
        "• YOUR specific subject on demand"
    ]

    for i, item in enumerate(dreambooth_items):
        color = '#388e3c' if '✓' in item or 'YOUR' in item else 'black'
        ax.text(8.5, 7 - i*0.6, item, fontsize=9, va='center', color=color)

    # Arrow connecting them
    ax.annotate('', xy=(8, 5.75), xytext=(6, 5.75),
                arrowprops=dict(arrowstyle='->', color='#757575', lw=3))
    ax.text(7, 6.2, "Builds On", fontsize=10, ha='center', fontweight='bold')

    # What's NEW box
    rect3 = FancyBboxPatch((4.5, 0.5), 5, 2, boxstyle="round,pad=0.1",
                            facecolor='#fff3e0', edgecolor='#ff9800', linewidth=2)
    ax.add_patch(rect3)
    ax.text(7, 2.1, "What's NEW in DreamBooth:", fontsize=10, fontweight='bold', ha='center')
    ax.text(7, 1.5, "1. Text conditioning (CLIP encoder)", fontsize=9, ha='center')
    ax.text(7, 1.0, "2. Subject token binding ('sks')", fontsize=9, ha='center')
    ax.text(7, 0.65, "3. Prior preservation loss", fontsize=9, ha='center')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "knowledge_transfer_ddpm.png", dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'knowledge_transfer_ddpm.png'}")


def main():
    """Generate all conceptual diagrams."""
    print("=" * 60)
    print("Generating Conceptual Diagrams for Module 12.5.1")
    print("=" * 60)

    print("\n1. DreamBooth Architecture Diagram...")
    create_dreambooth_architecture_diagram()

    print("\n2. Prior Preservation Diagram...")
    create_prior_preservation_diagram()

    print("\n3. TI vs LoRA Comparison...")
    create_ti_vs_lora_comparison()

    print("\n4. Knowledge Transfer from DDPM...")
    create_knowledge_transfer_diagram()

    print("\n" + "=" * 60)
    print(f"All diagrams saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
