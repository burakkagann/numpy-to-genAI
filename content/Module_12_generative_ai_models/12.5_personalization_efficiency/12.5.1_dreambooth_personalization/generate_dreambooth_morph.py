"""
DreamBooth Morphing Animation Generator

Creates a smooth 15-second GIF showing evolution through the learned
African fabric pattern latent space using spherical linear interpolation (SLERP).

The animation smoothly morphs between 5 keyframe patterns, demonstrating
the personalized LoRA model's ability to generate diverse, coherent fabric
designs while maintaining the learned "sks" subject characteristics.

This script follows the same pattern as Module 12.3.1's generate_ddpm_morph.py,
adapted for Stable Diffusion with LoRA personalization.

Usage:
    python generate_dreambooth_morph.py

Requirements:
    - Trained LoRA weights in models/fabric_lora/
    - diffusers, transformers, accelerate libraries
    - imageio for GIF creation

Author: NumPy-to-GenAI Project
"""

import os
import sys
from pathlib import Path

# Check for required libraries
try:
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    import torch
except ImportError as e:
    print(f"Error: Required library not found: {e}")
    print("Please install required packages:")
    print("  pip install diffusers transformers accelerate")
    sys.exit(1)

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import imageio.v2 as imageio
except ImportError:
    import imageio


# =============================================================================
# Configuration
# =============================================================================

# Model paths
MODULE_DIR = Path(__file__).parent
LORA_PATH = MODULE_DIR / "models" / "fabric_lora"
PRETRAINED_MODEL = "runwayml/stable-diffusion-v1-5"

# Animation parameters
NUM_KEYFRAMES = 5           # Number of distinct patterns to morph between
DURATION_SECONDS = 15       # Total animation duration
FPS = 30                    # Frames per second
OUTPUT_SIZE = 512           # Output resolution (native SD resolution)

# Generation parameters
PROMPT = "a sks african fabric pattern, detailed texture, high quality"
GUIDANCE_SCALE = 7.5        # Classifier-free guidance strength
NUM_INFERENCE_STEPS = 30    # Denoising steps per frame
RANDOM_SEED = 42            # For reproducible keyframes (change for different patterns)

# Output file
OUTPUT_PATH = MODULE_DIR / "dreambooth_fabric_morph.gif"


# =============================================================================
# SLERP Interpolation
# =============================================================================

def slerp(val, low, high):
    """
    Spherical linear interpolation between two tensors.

    SLERP produces smoother interpolations than linear interpolation
    for high-dimensional vectors like latents, as it interpolates along
    the surface of a hypersphere rather than cutting through it.

    This produces more perceptually smooth transitions in generative models.

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


def lerp(val, low, high):
    """Simple linear interpolation (fallback)."""
    return low * (1.0 - val) + high * val


# =============================================================================
# Model Loading
# =============================================================================

def load_pipeline(device='cuda'):
    """
    Load the Stable Diffusion pipeline with trained LoRA weights.

    Parameters:
        device: Computation device ('cuda' or 'cpu')

    Returns:
        pipe: StableDiffusionPipeline with LoRA loaded
    """
    print(f"Loading Stable Diffusion pipeline from: {PRETRAINED_MODEL}")

    # Load base pipeline with optimizations
    pipe = StableDiffusionPipeline.from_pretrained(
        PRETRAINED_MODEL,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )

    # Use faster scheduler for generation
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Load trained LoRA weights
    if LORA_PATH.exists():
        print(f"Loading LoRA weights from: {LORA_PATH}")
        pipe.load_lora_weights(str(LORA_PATH))
        print("LoRA weights loaded successfully!")
    else:
        print(f"Warning: LoRA weights not found at {LORA_PATH}")
        print("Generating with base model only (no personalization)")

    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    # Enable memory optimizations for long generation
    if device == 'cuda':
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass  # Not all versions support this

    return pipe


# =============================================================================
# Latent Generation
# =============================================================================

def generate_keyframe_latents(num_keyframes, shape, device, seed=None):
    """
    Generate random latent tensors for keyframes.

    Parameters:
        num_keyframes: Number of keyframe latents to generate
        shape: Shape of each latent tensor [1, C, H, W]
        device: Computation device
        seed: Random seed for reproducibility

    Returns:
        List of latent tensors
    """
    latents = []

    for i in range(num_keyframes):
        # Use different seed for each keyframe
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed + i * 1000)
        else:
            generator.manual_seed(i * 1000)

        latent = torch.randn(shape, generator=generator, device=device)
        latents.append(latent)

    return latents


def interpolate_latents(latent1, latent2, num_steps):
    """
    Generate SLERP-interpolated latents between two keyframes.

    Parameters:
        latent1: Starting latent
        latent2: Ending latent
        num_steps: Number of interpolation steps (excluding endpoints)

    Returns:
        List of interpolated latent tensors
    """
    interpolated = []
    for i in range(num_steps):
        t = i / num_steps  # Goes from 0 to just before 1
        interp_latent = slerp(t, latent1, latent2)
        interpolated.append(interp_latent)
    return interpolated


# =============================================================================
# Frame Generation
# =============================================================================

@torch.no_grad()
def generate_frame_from_latent(pipe, latent, prompt, device):
    """
    Generate an image from a specific latent tensor.

    Parameters:
        pipe: StableDiffusionPipeline
        latent: Starting latent tensor [1, 4, H, W]
        prompt: Text prompt for generation
        device: Computation device

    Returns:
        image: Generated image as numpy array [H, W, 3] in range [0, 255]
    """
    # Scale latent to match pipeline expectations
    latent = latent.to(device=device, dtype=pipe.unet.dtype)

    # Generate image from specific latent
    output = pipe(
        prompt=prompt,
        latents=latent,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        output_type="np"
    )

    # Convert to uint8 image
    image = output.images[0]
    image = (image * 255).clip(0, 255).astype(np.uint8)

    return image


# =============================================================================
# GIF Creation
# =============================================================================

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
    print("DreamBooth Fabric Pattern Morphing Animation Generator")
    print("=" * 60)

    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    if device == 'cpu':
        print("Warning: Generation on CPU will be very slow (1-2 hours)")
        print("Consider using a GPU for faster generation (~10-20 minutes)")

    # Check for LoRA weights
    if not LORA_PATH.exists():
        print(f"\nWarning: LoRA weights not found at '{LORA_PATH}'")
        print("The animation will use the base model without personalization.")
        print("\nTo use your trained model, ensure Exercise 3b has completed")
        print("and LoRA weights exist at: models/fabric_lora/")

    # Load pipeline
    pipe = load_pipeline(device)

    # Calculate frame counts
    total_frames = DURATION_SECONDS * FPS
    frames_per_transition = total_frames // NUM_KEYFRAMES
    actual_total_frames = frames_per_transition * NUM_KEYFRAMES

    print(f"\nAnimation parameters:")
    print(f"  - Duration: {DURATION_SECONDS} seconds")
    print(f"  - FPS: {FPS}")
    print(f"  - Keyframes: {NUM_KEYFRAMES}")
    print(f"  - Frames per transition: {frames_per_transition}")
    print(f"  - Total frames: {actual_total_frames}")
    print(f"  - Output size: {OUTPUT_SIZE}x{OUTPUT_SIZE}")
    print(f"  - Inference steps per frame: {NUM_INFERENCE_STEPS}")
    print(f"  - Prompt: '{PROMPT}'")
    print(f"  - Guidance scale: {GUIDANCE_SCALE}")

    # Latent dimensions for SD v1.5 (512x512 image -> 64x64 latent)
    latent_height = OUTPUT_SIZE // 8
    latent_width = OUTPUT_SIZE // 8
    latent_shape = (1, 4, latent_height, latent_width)

    # Generate keyframe latents
    print(f"\nGenerating {NUM_KEYFRAMES} keyframe latents (seed={RANDOM_SEED})...")
    keyframe_latents = generate_keyframe_latents(
        NUM_KEYFRAMES, latent_shape, device, seed=RANDOM_SEED
    )

    # Generate all interpolated latents
    print("Creating interpolated latent sequence for seamless loop...")
    all_latents = []

    for i in range(NUM_KEYFRAMES):
        # Get current and next keyframe (wrap around for seamless loop)
        current_latent = keyframe_latents[i]
        next_latent = keyframe_latents[(i + 1) % NUM_KEYFRAMES]

        # Interpolate between them
        interpolated = interpolate_latents(current_latent, next_latent, frames_per_transition)
        all_latents.extend(interpolated)

    print(f"Total latent vectors: {len(all_latents)}")

    # Generate frames
    print(f"\nGenerating {len(all_latents)} frames...")
    print("This may take 10-20 minutes on GPU, 1-2 hours on CPU...")
    frames = []

    for i, latent in enumerate(tqdm(all_latents, desc="Generating frames")):
        # Generate image from latent
        img = generate_frame_from_latent(pipe, latent, PROMPT, device)
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
    print("demonstrating the DreamBooth model's learned representation")
    print("of your personalized African fabric style.")
    print("\nThe 'sks' token binding ensures all generated patterns")
    print("maintain the characteristics learned from your training images.")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    generate_morph_animation()
