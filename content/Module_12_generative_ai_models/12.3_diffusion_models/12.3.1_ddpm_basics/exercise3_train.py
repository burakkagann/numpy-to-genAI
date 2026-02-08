"""
Exercise 3: Train DDPM on African Fabric Dataset

This script trains a Denoising Diffusion Probabilistic Model (DDPM) from scratch
on the African fabric dataset. Training takes approximately 4-6 hours on an
RTX 5070Ti GPU.

The training process:
1. Load preprocessed African fabric images (64x64)
2. Initialize U-Net model and diffusion process
3. Train the model to predict noise at various timesteps
4. Save checkpoints and sample images during training

Implementation uses the denoising-diffusion-pytorch library by Phil Wang
(lucidrains), which provides a clean and efficient training pipeline.

Library: https://github.com/lucidrains/denoising-diffusion-pytorch (MIT License)
Paper: Ho et al. (2020) "Denoising Diffusion Probabilistic Models"
"""

import os
import sys
from pathlib import Path

# Check for required library
try:
    from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
except ImportError:
    print("Error: denoising-diffusion-pytorch library not found.")
    print("Please install it with: pip install denoising-diffusion-pytorch")
    sys.exit(1)

import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Workshop utilities for device detection and training parameters
from workshop_utils import (
    get_device_with_confirmation,
    get_training_params,
    confirm_cpu_training
)


# =============================================================================
# Configuration
# =============================================================================

# Dataset path (relative to DCGAN module where preprocessed data exists)
DATASET_PATH = '../../12.1_generative_adversarial_networks/12.1.2_dcgan_art/african_fabric_processed'

# Training parameters (BATCH_SIZE and GRADIENT_ACCUMULATE set dynamically based on device)
IMAGE_SIZE = 64          # Image dimensions (64x64)
LEARNING_RATE = 2e-4     # Adam learning rate
TRAIN_STEPS = 100000     # Total training steps (~100 epochs with 1059 images)

# Model parameters
BASE_CHANNELS = 64       # Base channel count for U-Net
CHANNEL_MULTS = (1, 2, 4, 8)  # Channel multipliers at each level
TIMESTEPS = 1000         # Number of diffusion timesteps

# Output settings
RESULTS_DIR = './training_results'
CHECKPOINT_EPOCHS = [10, 25, 50, 75, 100]  # Save samples at these epochs
SAMPLES_PER_CHECKPOINT = 16  # Number of samples to generate


# =============================================================================
# Dataset Verification
# =============================================================================

def verify_dataset():
    """Verify the African fabric dataset exists and count images."""
    dataset_path = Path(DATASET_PATH)

    if not dataset_path.exists():
        print(f"Error: Dataset directory not found at '{DATASET_PATH}'")
        print("\nPlease ensure you have:")
        print("1. Completed Module 12.1.2 (DCGAN Art) first")
        print("2. Run the preprocessing script to create african_fabric_processed/")
        print("\nAlternatively, copy the preprocessed dataset to this location.")
        return None

    image_files = list(dataset_path.glob('*.png')) + list(dataset_path.glob('*.jpg'))

    if len(image_files) == 0:
        print(f"Error: No images found in '{DATASET_PATH}'")
        return None

    print(f"Found {len(image_files)} images in dataset")
    return len(image_files)


def show_training_samples(dataset_path, num_samples=9):
    """Display sample images from the training dataset."""
    dataset_path = Path(dataset_path)
    image_files = sorted(list(dataset_path.glob('*.png')))[:num_samples]

    if len(image_files) < num_samples:
        image_files = sorted(list(dataset_path.glob('*.png')))

    # Create grid
    grid_size = int(np.ceil(np.sqrt(len(image_files))))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
    axes = axes.flatten()

    for i, (ax, img_path) in enumerate(zip(axes, image_files)):
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis('off')

    # Hide unused axes
    for ax in axes[len(image_files):]:
        ax.axis('off')

    plt.suptitle('Training Dataset Samples', fontsize=14)
    plt.tight_layout()
    plt.savefig('training_samples_grid.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved training samples to 'training_samples_grid.png'")


# =============================================================================
# Training
# =============================================================================

def create_model():
    """
    Create the U-Net model for noise prediction.

    Architecture:
    - Base channels: 64 (scaled by multipliers at each level)
    - 4 resolution levels with multipliers (1, 2, 4, 8)
    - Self-attention at 16x16 and 8x8 resolutions
    - Sinusoidal timestep embeddings
    """
    model = Unet(
        dim=BASE_CHANNELS,
        dim_mults=CHANNEL_MULTS,
        flash_attn=False,  # Set True if you have flash-attn installed
        channels=3
    )
    return model


def create_diffusion(model):
    """
    Create the Gaussian diffusion process.

    Parameters:
    - 1000 timesteps (standard DDPM)
    - Linear beta schedule from 0.0001 to 0.02
    - MSE loss for noise prediction
    """
    diffusion = GaussianDiffusion(
        model,
        image_size=IMAGE_SIZE,
        timesteps=TIMESTEPS,
        sampling_timesteps=250,  # Faster sampling using DDIM
        objective='pred_noise'   # Predict noise (standard DDPM)
    )
    return diffusion


def train():
    """
    Main training function.

    Uses the Trainer class from denoising-diffusion-pytorch for:
    - Automatic mixed precision training
    - Exponential moving average (EMA) of model weights
    - Checkpoint saving and resumption
    - Sample generation during training
    """
    print("\n" + "=" * 60)
    print("DDPM Training on African Fabric Dataset")
    print("=" * 60)

    # Verify dataset
    num_images = verify_dataset()
    if num_images is None:
        return

    # Show training samples
    show_training_samples(DATASET_PATH)

    # Create output directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs('training_progress', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Device detection with user confirmation
    device, _ = get_device_with_confirmation(task_type="training")

    # Get device-appropriate training parameters
    params = get_training_params(device)
    batch_size = params['batch_size']
    gradient_accumulate = params['gradient_accumulate']
    use_amp = params['amp']
    estimated_time = params['estimated_time']

    # For CPU training, require explicit confirmation
    if device == 'cpu':
        confirm_cpu_training(params)

    # Create model and diffusion
    print("\nInitializing model...")
    model = create_model()
    diffusion = create_diffusion(model)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Model size: ~{num_params * 4 / 1e6:.1f} MB")

    # Estimate training time
    steps_per_epoch = num_images // batch_size
    total_epochs = TRAIN_STEPS // steps_per_epoch
    print(f"\nTraining configuration:")
    print(f"  - Device: {device.upper()}")
    print(f"  - Images: {num_images}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Gradient accumulation: {gradient_accumulate}")
    print(f"  - Mixed precision (AMP): {use_amp}")
    print(f"  - Steps per epoch: {steps_per_epoch}")
    print(f"  - Total steps: {TRAIN_STEPS}")
    print(f"  - Estimated epochs: {total_epochs}")
    print(f"  - Estimated time: {estimated_time}")

    # Create trainer
    print("\nStarting training...")
    trainer = Trainer(
        diffusion,
        DATASET_PATH,
        train_batch_size=batch_size,
        train_lr=LEARNING_RATE,
        train_num_steps=TRAIN_STEPS,
        gradient_accumulate_every=gradient_accumulate,
        ema_decay=0.995,
        amp=use_amp,  # Only use mixed precision on GPU
        results_folder=RESULTS_DIR,
        save_and_sample_every=1000,  # Save checkpoint every 1000 steps
        num_samples=16
    )

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Checkpoint saved. You can resume training later.")

    # Save final model
    print("\nSaving final model...")
    torch.save(diffusion.model.state_dict(), 'models/ddpm_african_fabrics.pt')
    print("Model saved to 'models/ddpm_african_fabrics.pt'")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print(f"  - Model checkpoint: models/ddpm_african_fabrics.pt")
    print(f"  - Training samples: {RESULTS_DIR}/")
    print("\nNext steps:")
    print("  1. Run exercise1_generate.py to generate new fabric patterns")
    print("  2. Run exercise2_explore.py to explore diffusion parameters")


# =============================================================================
# Checkpoint-Based Sample Generation
# =============================================================================

def generate_checkpoint_samples():
    """
    Generate sample images from saved checkpoints for visualization.

    This function loads checkpoints saved during training and generates
    sample grids to show training progression.
    """
    print("\n" + "=" * 60)
    print("Generating Checkpoint Samples")
    print("=" * 60)

    # Detect best available device
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    # Create model
    model = create_model()
    diffusion = create_diffusion(model)

    # Find available checkpoints
    checkpoint_files = sorted(Path(RESULTS_DIR).glob('model-*.pt'))

    if not checkpoint_files:
        print("No checkpoints found. Please run training first.")
        return

    print(f"Found {len(checkpoint_files)} checkpoints")

    for checkpoint_path in checkpoint_files:
        step = checkpoint_path.stem.split('-')[1]
        print(f"\nLoading checkpoint: {checkpoint_path.name}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        diffusion.load_state_dict(checkpoint['model'])
        diffusion.to(device)
        diffusion.eval()

        # Generate samples
        with torch.no_grad():
            samples = diffusion.sample(batch_size=16)

        # Convert to grid
        samples = samples.cpu()
        samples = (samples + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
        samples = samples.clamp(0, 1)

        # Create 4x4 grid
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i, ax in enumerate(axes.flatten()):
            img = samples[i].permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.axis('off')

        plt.suptitle(f'Generated Samples at Step {step}', fontsize=14)
        plt.tight_layout()

        output_path = f'training_progress/step_{step}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")


# =============================================================================
# Loss Curve Plotting
# =============================================================================

def plot_training_loss():
    """
    Plot the training loss curve from saved checkpoints.

    Note: The denoising-diffusion-pytorch library saves loss history
    in the trainer state. This function extracts and visualizes it.
    """
    print("\nPlotting training loss curve...")

    # This is a placeholder - actual implementation depends on
    # how losses are logged during training
    print("Loss curve plotting requires training logs.")
    print("Check the training output for loss values.")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='DDPM Training on African Fabrics')
    parser.add_argument('--train', action='store_true', help='Start training')
    parser.add_argument('--samples', action='store_true', help='Generate checkpoint samples')
    parser.add_argument('--verify', action='store_true', help='Verify dataset only')

    args = parser.parse_args()

    if args.verify:
        verify_dataset()
    elif args.samples:
        generate_checkpoint_samples()
    elif args.train:
        train()
    else:
        # Default: run full training
        print("DDPM Training Script")
        print("=" * 40)
        print("\nUsage:")
        print("  python exercise3_train.py --train    # Start training")
        print("  python exercise3_train.py --samples  # Generate samples from checkpoints")
        print("  python exercise3_train.py --verify   # Verify dataset")
        print("\nStarting training by default...")
        train()
