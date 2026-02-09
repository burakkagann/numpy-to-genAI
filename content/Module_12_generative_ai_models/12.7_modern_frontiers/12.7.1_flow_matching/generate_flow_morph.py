"""
Flow Matching Morphing Animation Generator

Creates a smooth 15-second GIF showing evolution through the learned
African fabric pattern space using spherical linear interpolation (SLERP).

The animation smoothly morphs between 5 keyframe patterns, demonstrating
the model's ability to generate diverse, coherent fabric designs.

Usage:
    python generate_flow_morph.py

Requirements:
    - Trained model checkpoint (models/flow_matching_fabrics.pt)
    - imageio for GIF creation

Key Features:
    - SLERP interpolation in noise space for smooth transitions
    - ODE integration with only 50 steps (vs DDPM's 250)
    - Seamless looping animation
    - 256x256 output (upscaled from 64x64)

Inspired by:
- Meta AI Flow Matching: https://github.com/facebookresearch/flow_matching
- Lipman et al. (2023) "Flow Matching for Generative Modeling"
- Cambridge MLG Blog: https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html

Author: NumPy-to-GenAI Project
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import imageio.v2 as imageio
except ImportError:
    import imageio

# Import our Flow Matching model
try:
    from flow_model import SimpleFlowUNet, count_parameters
except ImportError:
    print("Error: flow_model.py not found in the same directory.")
    print("Please ensure flow_model.py exists alongside this script.")
    sys.exit(1)


# =============================================================================
# Configuration
# =============================================================================

# Model checkpoint path
MODEL_PATH = SCRIPT_DIR / 'models' / 'flow_matching_fabrics.pt'

# Model architecture (must match training configuration)
IMAGE_SIZE = 64
BASE_CHANNELS = 64

# Animation parameters (matching DDPM for consistency)
NUM_KEYFRAMES = 5           # Number of distinct patterns to morph between
DURATION_SECONDS = 15       # Total animation duration
FPS = 30                    # Frames per second
OUTPUT_SIZE = 256           # Output resolution (upscaled from 64x64)

# Sampling parameters
SAMPLING_STEPS = 50         # ODE integration steps (Flow Matching is efficient!)
RANDOM_SEED = 42            # For reproducible keyframes

# Output file
OUTPUT_PATH = SCRIPT_DIR / 'flow_morph.gif'


# =============================================================================
# SLERP Interpolation
# =============================================================================

def slerp(val, low, high):
    """
    Spherical linear interpolation between two tensors.

    SLERP produces smoother interpolations than linear interpolation
    for high-dimensional vectors like noise, as it interpolates along
    the surface of a hypersphere rather than cutting through it.

    Parameters:
        val: Interpolation factor (0.0 = low, 1.0 = high)
        low: Starting tensor
        high: Ending tensor

    Returns:
        Interpolated tensor
    """
    # Flatten tensors for dot product
    low_flat = low.flatten()
    high_flat = high.flatten()

    # Normalize vectors
    low_norm = low_flat / torch.norm(low_flat)
    high_norm = high_flat / torch.norm(high_flat)

    # Compute angle between vectors
    dot = torch.clamp(torch.dot(low_norm, high_norm), -1.0, 1.0)
    omega = torch.acos(dot)

    # Handle near-parallel vectors (fall back to linear interpolation)
    if torch.abs(omega) < 1e-10:
        return low * (1.0 - val) + high * val

    # SLERP formula
    so = torch.sin(omega)
    result = (torch.sin((1.0 - val) * omega) / so) * low_flat + \
             (torch.sin(val * omega) / so) * high_flat

    return result.view(low.shape)


# =============================================================================
# Model Loading
# =============================================================================

def load_model(model_path, device='cpu'):
    """
    Load the trained Flow Matching model.

    Parameters:
        model_path: Path to saved model weights
        device: Device to load model on

    Returns:
        model: Loaded SimpleFlowUNet model
    """
    # Create model architecture
    model = SimpleFlowUNet(
        in_channels=3,
        base_channels=BASE_CHANNELS
    )

    print(f"Model parameters: {count_parameters(model):,}")

    # Load weights
    if os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        print("Model loaded successfully!")
    else:
        raise FileNotFoundError(f"Model not found at '{model_path}'")

    model.to(device)
    model.eval()

    return model


# =============================================================================
# Frame Generation (ODE Integration)
# =============================================================================

@torch.no_grad()
def generate_from_noise(model, noise, num_steps, device):
    """
    Generate an image from a specific noise tensor using ODE integration.

    This is the core Flow Matching sampling: integrate the learned velocity
    field from t=0 (noise) to t=1 (data) using Euler method.

    Parameters:
        model: Trained velocity network v(x, t)
        noise: Starting noise tensor [1, 3, H, W]
        num_steps: ODE integration steps
        device: Computation device

    Returns:
        image: Generated image as numpy array [H, W, 3] in range [0, 255]
    """
    model.eval()

    # Start from provided noise
    x = noise.to(device)
    batch_size = x.shape[0]

    # Time step size
    dt = 1.0 / num_steps

    # Euler integration from t=0 to t=1
    for i in range(num_steps):
        t = i / num_steps
        t_batch = torch.full((batch_size,), t, device=device)

        # Get velocity at current position and time
        v = model(x, t_batch)

        # Euler step: move in direction of velocity
        x = x + dt * v

    # Clamp to valid range
    x = x.clamp(-1, 1)

    # Convert to numpy image
    img = x.cpu()
    img = (img + 1) / 2  # [-1, 1] -> [0, 1]
    img = img.clamp(0, 1)
    img = (img[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    return img


def generate_keyframe_noises(num_keyframes, shape, seed=None):
    """
    Generate random noise tensors for keyframes.

    Parameters:
        num_keyframes: Number of keyframe noises to generate
        shape: Shape of each noise tensor [1, C, H, W]
        seed: Random seed for reproducibility

    Returns:
        List of noise tensors
    """
    if seed is not None:
        torch.manual_seed(seed)

    noises = [torch.randn(shape) for _ in range(num_keyframes)]
    return noises


def interpolate_noises(noise1, noise2, num_steps):
    """
    Generate interpolated noise tensors between two keyframes using SLERP.

    Parameters:
        noise1: Starting noise
        noise2: Ending noise
        num_steps: Number of interpolation steps (excluding endpoints)

    Returns:
        List of interpolated noise tensors
    """
    interpolated = []
    for i in range(num_steps):
        t = i / num_steps  # Goes from 0 to just before 1
        interp_noise = slerp(t, noise1, noise2)
        interpolated.append(interp_noise)
    return interpolated


# =============================================================================
# GIF Creation
# =============================================================================

def upscale_image(img, target_size):
    """
    Upscale image using high-quality bicubic interpolation.

    Parameters:
        img: Image as numpy array [H, W, 3]
        target_size: Target dimension (square)

    Returns:
        Upscaled image as numpy array
    """
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((target_size, target_size), Image.BICUBIC)
    return np.array(pil_img)


def create_gif(frames, output_path, fps):
    """
    Create a smooth GIF from frames.

    Parameters:
        frames: List of numpy arrays [H, W, 3]
        output_path: Output file path
        fps: Frames per second
    """
    print(f"\nCreating GIF with {len(frames)} frames at {fps} FPS...")

    # Save GIF
    imageio.mimsave(
        str(output_path),
        frames,
        fps=fps,
        loop=0  # Infinite loop
    )

    # Get file size
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved to: {output_path}")
    print(f"File size: {file_size:.2f} MB")
    print(f"Duration: {len(frames) / fps:.1f} seconds")


# =============================================================================
# Main Generation
# =============================================================================

def generate_morph_animation():
    """Main function to generate the morphing animation."""
    print("=" * 60)
    print("Flow Matching Fabric Pattern Morphing Animation Generator")
    print("=" * 60)

    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    if device == 'cpu':
        print("Warning: Generation on CPU will be slow (~10-20 minutes)")
        print("Consider using a GPU for faster generation (~3-5 minutes)")

    # Check for checkpoint
    if not os.path.exists(MODEL_PATH):
        print(f"\nError: Checkpoint not found at '{MODEL_PATH}'")
        print("Please ensure training has completed (exercise3_train.py)")
        return

    # Load model
    model = load_model(MODEL_PATH, device)

    # Calculate frame counts
    total_frames = DURATION_SECONDS * FPS  # 450 frames
    frames_per_transition = total_frames // NUM_KEYFRAMES  # 90 frames per transition
    actual_total_frames = frames_per_transition * NUM_KEYFRAMES  # 450 frames

    print(f"\nAnimation parameters:")
    print(f"  - Duration: {DURATION_SECONDS} seconds")
    print(f"  - FPS: {FPS}")
    print(f"  - Keyframes: {NUM_KEYFRAMES}")
    print(f"  - Frames per transition: {frames_per_transition}")
    print(f"  - Total frames: {actual_total_frames}")
    print(f"  - Output size: {OUTPUT_SIZE}x{OUTPUT_SIZE}")
    print(f"  - ODE sampling steps: {SAMPLING_STEPS}")

    # Generate keyframe noises
    print(f"\nGenerating {NUM_KEYFRAMES} keyframe noise vectors (seed={RANDOM_SEED})...")
    noise_shape = (1, 3, IMAGE_SIZE, IMAGE_SIZE)
    keyframe_noises = generate_keyframe_noises(NUM_KEYFRAMES, noise_shape, seed=RANDOM_SEED)

    # Generate all interpolated noises for seamless loop
    print("Creating interpolated noise sequence for seamless loop...")
    all_noises = []

    for i in range(NUM_KEYFRAMES):
        # Get current and next keyframe (wrap around for seamless loop)
        current_noise = keyframe_noises[i]
        next_noise = keyframe_noises[(i + 1) % NUM_KEYFRAMES]

        # Interpolate between them
        interpolated = interpolate_noises(current_noise, next_noise, frames_per_transition)
        all_noises.extend(interpolated)

    print(f"Total noise vectors: {len(all_noises)}")

    # Generate frames
    print(f"\nGenerating {len(all_noises)} frames...")
    frames = []

    for i, noise in enumerate(tqdm(all_noises, desc="Generating frames")):
        # Generate image from noise via ODE integration
        img = generate_from_noise(model, noise, SAMPLING_STEPS, device)

        # Upscale to output size
        if OUTPUT_SIZE != IMAGE_SIZE:
            img = upscale_image(img, OUTPUT_SIZE)

        frames.append(img)

        # Clear GPU cache periodically
        if device == 'cuda' and i % 50 == 0:
            torch.cuda.empty_cache()

    # Create GIF
    create_gif(frames, OUTPUT_PATH, FPS)

    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    print(f"\nOutput: {OUTPUT_PATH}")
    print("\nThe animation shows smooth morphing between fabric patterns,")
    print("demonstrating the Flow Matching model's learned representation space.")
    print(f"\nKey advantage: Only {SAMPLING_STEPS} ODE steps per frame (vs 250 for DDPM)!")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    generate_morph_animation()
