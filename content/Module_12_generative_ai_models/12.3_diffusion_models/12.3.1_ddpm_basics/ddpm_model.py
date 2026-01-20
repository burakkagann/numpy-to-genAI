"""
U-Net Architecture for Denoising Diffusion Probabilistic Models (DDPM)

This module implements a U-Net architecture specifically designed for diffusion models.
The network learns to predict the noise added to images at each timestep during
the forward diffusion process.

Key architectural components:
1. Sinusoidal timestep embeddings to condition the network on diffusion time
2. Residual blocks with GroupNorm for stable training
3. Self-attention layers at intermediate resolutions for global context
4. Skip connections between encoder and decoder for preserving spatial details

Implementation inspired by:
- Ho et al. (2020) "Denoising Diffusion Probabilistic Models"
- lucidrains/denoising-diffusion-pytorch (MIT License)
- Ronneberger et al. (2015) "U-Net" architecture
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Timestep Embedding
# =============================================================================

class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal position embedding for timesteps.

    Creates time-dependent embeddings using sine and cosine functions at
    different frequencies. This allows the network to distinguish between
    different noise levels during the diffusion process.

    The embedding formula follows the transformer positional encoding:
    PE(t, 2i) = sin(t / 10000^(2i/d))
    PE(t, 2i+1) = cos(t / 10000^(2i/d))
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2

        # Compute frequency scaling factors
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)

        # Apply sine and cosine to get final embedding
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        return embeddings


# =============================================================================
# Building Blocks
# =============================================================================

class Block(nn.Module):
    """
    Basic convolutional block with GroupNorm activation.

    Structure: Conv2d -> GroupNorm -> SiLU activation
    GroupNorm is preferred over BatchNorm for diffusion models as it
    performs better with small batch sizes.
    """

    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.activation = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class ResidualBlock(nn.Module):
    """
    Residual block with timestep conditioning.

    The block processes features through two convolutions while adding
    timestep information via a learned projection. This allows the network
    to adapt its behavior based on the current noise level.

    Structure:
    Input -> Block1 -> (+ time_embedding) -> Block2 -> (+ residual) -> Output
    """

    def __init__(self, in_channels, out_channels, time_emb_dim, groups=8):
        super().__init__()

        self.block1 = Block(in_channels, out_channels, groups)
        self.block2 = Block(out_channels, out_channels, groups)

        # Project timestep embedding to match feature channels
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        # Residual connection (with channel adjustment if needed)
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, time_emb):
        # First convolution block
        h = self.block1(x)

        # Add timestep embedding (broadcast across spatial dimensions)
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]

        # Second convolution block
        h = self.block2(h)

        # Residual connection
        return h + self.residual_conv(x)


class SelfAttention(nn.Module):
    """
    Self-attention layer for capturing global context.

    Allows the network to relate different spatial positions, which is
    important for generating coherent patterns. Applied at intermediate
    resolutions (16x16, 8x8) where computational cost is manageable.
    """

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        batch, channels, height, width = x.shape

        # Normalize input
        h = self.norm(x)

        # Compute query, key, value projections
        qkv = self.qkv(h)
        qkv = qkv.reshape(batch, 3, self.num_heads, self.head_dim, height * width)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # Attention computation
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) * self.scale
        attn = attn.softmax(dim=-1)

        # Apply attention to values
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)
        out = out.reshape(batch, channels, height, width)

        # Project and add residual
        return x + self.proj(out)


class Downsample(nn.Module):
    """Spatial downsampling using strided convolution."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Spatial upsampling using interpolation + convolution."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


# =============================================================================
# U-Net Architecture
# =============================================================================

class UNet(nn.Module):
    """
    U-Net architecture for diffusion models.

    The network follows an encoder-decoder structure with skip connections.
    Feature maps are progressively downsampled in the encoder, processed
    at a bottleneck, then upsampled in the decoder while incorporating
    encoder features via skip connections.

    Architecture for 64x64 images:

    Encoder:
        64x64 -> [ResBlock, ResBlock, Attn, Down] -> 32x32
        32x32 -> [ResBlock, ResBlock, Attn, Down] -> 16x16
        16x16 -> [ResBlock, ResBlock, Attn, Down] -> 8x8

    Bottleneck:
        8x8 -> [ResBlock, Attn, ResBlock]

    Decoder:
        8x8 -> [Up, ResBlock, ResBlock, Attn] -> 16x16
        16x16 -> [Up, ResBlock, ResBlock, Attn] -> 32x32
        32x32 -> [Up, ResBlock, ResBlock, Attn] -> 64x64

    Parameters:
        image_channels: Number of input/output channels (3 for RGB)
        base_channels: Base channel count (multiplied at each level)
        channel_multipliers: Channel multiplier at each resolution level
        num_res_blocks: Number of residual blocks per level
        attention_levels: Which levels include self-attention (0-indexed)
    """

    def __init__(
        self,
        image_channels=3,
        base_channels=64,
        channel_multipliers=(1, 2, 4, 8),
        num_res_blocks=2,
        attention_levels=(1, 2, 3)
    ):
        super().__init__()

        self.image_channels = image_channels
        self.base_channels = base_channels

        # Timestep embedding dimension
        time_dim = base_channels * 4

        # Timestep embedding layers
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # Initial convolution
        self.init_conv = nn.Conv2d(image_channels, base_channels, kernel_size=3, padding=1)

        # Calculate channel sizes at each level
        channels = [base_channels]
        for mult in channel_multipliers:
            channels.append(base_channels * mult)

        # Build encoder (downsampling path)
        self.encoder = nn.ModuleList()
        self.encoder_attns = nn.ModuleList()

        in_ch = base_channels
        for level, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult

            # Residual blocks at this level
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_blocks.append(ResidualBlock(in_ch, out_ch, time_dim))
                in_ch = out_ch
            self.encoder.append(level_blocks)

            # Attention at this level
            if level in attention_levels:
                self.encoder_attns.append(SelfAttention(out_ch))
            else:
                self.encoder_attns.append(nn.Identity())

            # Downsample (except at last level)
            if level < len(channel_multipliers) - 1:
                self.encoder.append(nn.ModuleList([Downsample(out_ch)]))

        # Bottleneck
        mid_ch = channels[-1]
        self.bottleneck = nn.ModuleList([
            ResidualBlock(mid_ch, mid_ch, time_dim),
            SelfAttention(mid_ch),
            ResidualBlock(mid_ch, mid_ch, time_dim)
        ])

        # Build decoder (upsampling path)
        self.decoder = nn.ModuleList()
        self.decoder_attns = nn.ModuleList()

        for level, mult in enumerate(reversed(channel_multipliers)):
            out_ch = base_channels * mult

            # Upsample (except at first level)
            if level > 0:
                prev_ch = channels[len(channel_multipliers) - level]
                self.decoder.append(nn.ModuleList([Upsample(prev_ch)]))

            # Residual blocks (with skip connection, hence 2x channels)
            level_blocks = nn.ModuleList()
            for i in range(num_res_blocks + 1):
                skip_ch = channels[len(channel_multipliers) - level] if i == 0 else out_ch
                level_blocks.append(ResidualBlock(in_ch + skip_ch, out_ch, time_dim))
                in_ch = out_ch
            self.decoder.append(level_blocks)

            # Attention at this level
            rev_level = len(channel_multipliers) - 1 - level
            if rev_level in attention_levels:
                self.decoder_attns.append(SelfAttention(out_ch))
            else:
                self.decoder_attns.append(nn.Identity())

        # Final output layers
        self.final_block = Block(base_channels, base_channels)
        self.final_conv = nn.Conv2d(base_channels, image_channels, kernel_size=1)

    def forward(self, x, time):
        """
        Forward pass of the U-Net.

        Parameters:
            x: Noisy image tensor [batch, channels, height, width]
            time: Timestep tensor [batch]

        Returns:
            Predicted noise tensor [batch, channels, height, width]
        """
        # Timestep embedding
        t = self.time_embedding(time)

        # Initial convolution
        x = self.init_conv(x)

        # Store skip connections
        skips = [x]

        # Encoder path
        block_idx = 0
        for i, (blocks, attn) in enumerate(zip(self.encoder, self.encoder_attns)):
            if isinstance(blocks[0], Downsample):
                x = blocks[0](x)
                skips.append(x)
            else:
                for block in blocks:
                    x = block(x, t)
                x = attn(x)
                skips.append(x)

        # Bottleneck
        x = self.bottleneck[0](x, t)
        x = self.bottleneck[1](x)
        x = self.bottleneck[2](x, t)

        # Decoder path
        for i, (blocks, attn) in enumerate(zip(self.decoder, self.decoder_attns)):
            if isinstance(blocks[0], Upsample):
                x = blocks[0](x)
            else:
                for j, block in enumerate(blocks):
                    skip = skips.pop()
                    x = torch.cat([x, skip], dim=1)
                    x = block(x, t)
                x = attn(x)

        # Final output
        x = self.final_block(x)
        x = self.final_conv(x)

        return x


def count_parameters(model):
    """Count the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    # Test the model with random input
    print("Testing U-Net Architecture")
    print("=" * 50)

    # Create model
    model = UNet(
        image_channels=3,
        base_channels=64,
        channel_multipliers=(1, 2, 4, 8),
        num_res_blocks=2,
        attention_levels=(1, 2, 3)
    )

    # Print parameter count
    params = count_parameters(model)
    print(f"Total parameters: {params:,}")
    print(f"Model size: ~{params * 4 / 1e6:.1f} MB (float32)")

    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 3, 64, 64)
    t = torch.randint(0, 1000, (batch_size,))

    print(f"\nInput shape: {x.shape}")
    print(f"Timesteps: {t.tolist()}")

    with torch.no_grad():
        output = model(x, t)

    print(f"Output shape: {output.shape}")
    print("\nModel test passed!")
