"""
Diffusion Process for Denoising Diffusion Probabilistic Models (DDPM)

This module implements the forward and reverse diffusion processes that form
the core of DDPM. The diffusion model learns to reverse a gradual noising
process, enabling generation of images from pure noise.

Key concepts:
1. Forward process: Gradually adds noise to images over T timesteps
2. Reverse process: Neural network learns to denoise step by step
3. Noise schedule: Controls how much noise is added at each step

Mathematical formulation based on:
- Ho et al. (2020) "Denoising Diffusion Probabilistic Models"
- LabML.ai DDPM implementation (educational reference)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion Process for image generation.

    The diffusion process consists of:

    Forward process q(x_t | x_{t-1}):
        Gradually adds Gaussian noise to data over T timesteps.
        At timestep T, the data becomes approximately pure noise.

    Reverse process p(x_{t-1} | x_t):
        A neural network learns to reverse the forward process,
        predicting the noise added at each step.

    Key equations:
        Forward: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        Reverse: x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1-alpha_bar_t)) * pred_noise) + sigma_t * z

    Parameters:
        model: U-Net model for noise prediction
        timesteps: Number of diffusion steps (default: 1000)
        beta_start: Starting value of variance schedule
        beta_end: Ending value of variance schedule
        schedule_type: 'linear' or 'cosine' noise schedule
    """

    def __init__(
        self,
        model,
        timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        schedule_type='linear'
    ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        # Create noise schedule
        if schedule_type == 'linear':
            betas = self._linear_schedule(timesteps, beta_start, beta_end)
        elif schedule_type == 'cosine':
            betas = self._cosine_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        # Pre-compute diffusion constants
        # These are used throughout the forward and reverse process
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register as buffers (will be moved to correct device automatically)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # For forward process sampling
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

        # For reverse process sampling
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('posterior_variance', betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

    def _linear_schedule(self, timesteps, beta_start, beta_end):
        """
        Linear noise schedule.

        Linearly interpolates beta values from start to end.
        This is the original schedule from Ho et al. (2020).
        """
        return torch.linspace(beta_start, beta_end, timesteps)

    def _cosine_schedule(self, timesteps, s=0.008):
        """
        Cosine noise schedule.

        Uses cosine function to create a smoother schedule that
        preserves more image information in early timesteps.
        From Nichol & Dhariwal (2021) "Improved DDPM".
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def _extract(self, tensor, t, shape):
        """Extract values from tensor at timestep t and reshape for broadcasting."""
        batch_size = t.shape[0]
        out = tensor.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(shape) - 1)))

    # =========================================================================
    # Forward Process
    # =========================================================================

    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion process: add noise to image at timestep t.

        Uses the closed-form formula to directly sample x_t from x_0:
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise

        This allows sampling any timestep directly without iterating.

        Parameters:
            x_0: Original clean images [batch, channels, height, width]
            t: Timesteps to sample [batch]
            noise: Optional pre-generated noise

        Returns:
            x_t: Noisy images at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def visualize_forward_process(self, x_0, steps=None):
        """
        Visualize the forward diffusion process at multiple timesteps.

        Useful for understanding how noise is progressively added.

        Parameters:
            x_0: Single clean image [1, channels, height, width]
            steps: List of timesteps to visualize (default: evenly spaced)

        Returns:
            List of (timestep, noisy_image) tuples
        """
        if steps is None:
            steps = [0, 250, 500, 750, 999]

        results = []
        for t in steps:
            t_tensor = torch.tensor([t], device=x_0.device)
            x_t = self.q_sample(x_0, t_tensor)
            results.append((t, x_t))

        return results

    # =========================================================================
    # Reverse Process (Sampling)
    # =========================================================================

    @torch.no_grad()
    def p_sample(self, x_t, t):
        """
        Single reverse diffusion step: denoise x_t to get x_{t-1}.

        Uses the trained model to predict noise, then computes the
        mean of p(x_{t-1} | x_t) and adds appropriate noise.

        Parameters:
            x_t: Noisy images at timestep t [batch, channels, height, width]
            t: Current timestep [batch]

        Returns:
            x_{t-1}: Slightly less noisy images
        """
        # Predict noise using the model
        pred_noise = self.model(x_t, t)

        # Extract precomputed values for this timestep
        betas_t = self._extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x_t.shape)

        # Compute predicted mean
        model_mean = sqrt_recip_alphas_t * (x_t - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t)

        # Add noise (except at t=0)
        if t[0] > 0:
            noise = torch.randn_like(x_t)
            posterior_variance_t = self._extract(self.posterior_variance, t, x_t.shape)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
        else:
            return model_mean

    @torch.no_grad()
    def sample(self, batch_size, image_size, channels=3, device='cpu', return_intermediates=False):
        """
        Generate images by running the full reverse diffusion process.

        Starting from pure Gaussian noise, iteratively denoise for T steps
        to generate clean images.

        Parameters:
            batch_size: Number of images to generate
            image_size: Size of images (height = width)
            channels: Number of image channels
            device: Device to generate on
            return_intermediates: If True, return intermediate images

        Returns:
            Generated images [batch, channels, height, width]
            (optionally) List of intermediate images
        """
        shape = (batch_size, channels, image_size, image_size)

        # Start from pure noise
        x = torch.randn(shape, device=device)

        intermediates = []

        # Reverse diffusion: denoise step by step
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch)

            if return_intermediates and t % 100 == 0:
                intermediates.append((t, x.clone()))

        if return_intermediates:
            intermediates.append((0, x))
            return x, intermediates
        return x

    @torch.no_grad()
    def sample_with_steps(self, batch_size, image_size, num_steps=None, channels=3, device='cpu'):
        """
        Generate images using a reduced number of sampling steps.

        Uses evenly spaced timesteps for faster generation.
        Note: This is a simplified version; DDIM provides better quality
        with fewer steps.

        Parameters:
            batch_size: Number of images to generate
            image_size: Size of images
            num_steps: Number of sampling steps (default: full timesteps)
            channels: Number of image channels
            device: Device to generate on

        Returns:
            Generated images
        """
        if num_steps is None:
            num_steps = self.timesteps

        shape = (batch_size, channels, image_size, image_size)
        x = torch.randn(shape, device=device)

        # Select evenly spaced timesteps
        step_size = self.timesteps // num_steps
        timesteps = list(range(self.timesteps - 1, 0, -step_size))

        for t in timesteps:
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch)

        return x

    # =========================================================================
    # Training
    # =========================================================================

    def compute_loss(self, x_0):
        """
        Compute training loss for a batch of images.

        The loss is the MSE between predicted and actual noise:
        L = E[||noise - model(x_t, t)||^2]

        Parameters:
            x_0: Clean training images [batch, channels, height, width]

        Returns:
            Loss value (scalar tensor)
        """
        batch_size = x_0.shape[0]
        device = x_0.device

        # Sample random timesteps for each image in batch
        t = torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)

        # Sample noise and create noisy images
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)

        # Predict noise
        pred_noise = self.model(x_t, t)

        # MSE loss between predicted and actual noise
        loss = F.mse_loss(pred_noise, noise)

        return loss


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    from ddpm_model import UNet

    print("Testing Gaussian Diffusion Process")
    print("=" * 50)

    # Create a simple model and diffusion process
    model = UNet(image_channels=3, base_channels=32, channel_multipliers=(1, 2, 4))
    diffusion = GaussianDiffusion(model, timesteps=1000)

    # Test forward process
    print("\n1. Testing forward process (adding noise)...")
    x_0 = torch.randn(2, 3, 64, 64)  # Fake clean images
    t = torch.tensor([100, 500])     # Different timesteps

    x_t = diffusion.q_sample(x_0, t)
    print(f"   Input shape: {x_0.shape}")
    print(f"   Output shape: {x_t.shape}")
    print(f"   Timesteps: {t.tolist()}")

    # Test loss computation
    print("\n2. Testing loss computation...")
    loss = diffusion.compute_loss(x_0)
    print(f"   Loss value: {loss.item():.4f}")

    # Test single reverse step
    print("\n3. Testing reverse step...")
    x_t = torch.randn(2, 3, 64, 64)
    t = torch.tensor([500, 500])
    x_prev = diffusion.p_sample(x_t, t)
    print(f"   Input shape: {x_t.shape}")
    print(f"   Output shape: {x_prev.shape}")

    # Test noise schedule visualization
    print("\n4. Noise schedule visualization...")
    print(f"   Beta range: [{diffusion.betas[0]:.6f}, {diffusion.betas[-1]:.6f}]")
    print(f"   Alpha_bar at t=0: {diffusion.alphas_cumprod[0]:.4f}")
    print(f"   Alpha_bar at t=500: {diffusion.alphas_cumprod[500]:.4f}")
    print(f"   Alpha_bar at t=999: {diffusion.alphas_cumprod[999]:.4f}")

    print("\nAll tests passed!")
