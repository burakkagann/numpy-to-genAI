"""
Exercise 3.4.4: Fourier Art - Concept Diagram

This script creates an educational diagram showing the complete FFT pipeline:
Original Image → FFT → Magnitude Spectrum → Apply Filter → Inverse FFT → Result

Author: Claude (NumPy-to-GenAI Project)
Date: 2025-12-07

Thesis Metadata:
    Framework: F1-Hands-On
    Module: Module_03_transformations_effects
    Exercise Type: Diagram
    Cognitive Load: Medium
    New Concepts: FFT pipeline, frequency filtering workflow
    Prerequisites: Basic FFT understanding
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

def create_synthetic_image(size=128):
    """Create a synthetic image with interesting frequency content."""
    image = np.zeros((size, size), dtype=np.float64)

    # Add concentric circles (creates radial frequency pattern)
    center = size // 2
    for y in range(size):
        for x in range(size):
            dist = np.sqrt((x - center)**2 + (y - center)**2)
            # Create rings with varying frequency
            image[y, x] = 128 + 127 * np.sin(dist * 0.3)

    return image

def apply_fft(image):
    """Apply FFT and return shifted result."""
    fft_result = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft_result)
    return fft_shifted

def get_magnitude_spectrum(fft_data):
    """Get log-scaled magnitude spectrum for visualization."""
    magnitude = np.abs(fft_data)
    magnitude_log = np.log1p(magnitude)
    normalized = (magnitude_log / magnitude_log.max() * 255).astype(np.uint8)
    return normalized

def create_circular_mask(size, radius, mask_type='low'):
    """Create a circular mask for frequency filtering."""
    mask = np.zeros((size, size), dtype=np.float64)
    center = size // 2

    for y in range(size):
        for x in range(size):
            dist = np.sqrt((x - center)**2 + (y - center)**2)
            if mask_type == 'low':
                mask[y, x] = 1.0 if dist <= radius else 0.0
            else:  # high-pass
                mask[y, x] = 0.0 if dist <= radius else 1.0

    return mask

def apply_filter_and_reconstruct(fft_shifted, mask):
    """Apply frequency filter and reconstruct image."""
    filtered = fft_shifted * mask
    fft_unshifted = np.fft.ifftshift(filtered)
    reconstructed = np.fft.ifft2(fft_unshifted)
    return np.abs(reconstructed)

# Create the pipeline demonstration
size = 128
padding = 10
label_height = 25

# Step 1: Create original image
original = create_synthetic_image(size)

# Step 2: Apply FFT
fft_shifted = apply_fft(original)
magnitude = get_magnitude_spectrum(fft_shifted)

# Step 3: Create low-pass filter mask
mask = create_circular_mask(size, radius=20, mask_type='low')
mask_display = (mask * 255).astype(np.uint8)

# Step 4: Apply filter (visualize filtered spectrum)
filtered_fft = fft_shifted * mask
filtered_magnitude = get_magnitude_spectrum(filtered_fft)

# Step 5: Reconstruct (inverse FFT)
reconstructed = apply_filter_and_reconstruct(fft_shifted, mask)
reconstructed_normalized = (reconstructed / reconstructed.max() * 255).astype(np.uint8)

# Create the diagram layout (5 images in a row with labels)
num_images = 5
diagram_width = num_images * size + (num_images - 1) * padding
diagram_height = size + label_height + padding

# Create output array
diagram = np.ones((diagram_height, diagram_width), dtype=np.uint8) * 240  # Light gray background

# Place images
images = [
    (original.astype(np.uint8), "Original"),
    (magnitude, "FFT Spectrum"),
    (mask_display, "Low-Pass Mask"),
    (filtered_magnitude, "Filtered"),
    (reconstructed_normalized, "Result (Blurred)")
]

for i, (img, label) in enumerate(images):
    x_start = i * (size + padding)
    diagram[label_height:label_height + size, x_start:x_start + size] = img

# Save the diagram
result = Image.fromarray(diagram, mode='L')

# Add labels using PIL
draw = ImageDraw.Draw(result)
try:
    font = ImageFont.truetype("arial.ttf", 12)
except:
    font = ImageFont.load_default()

for i, (_, label) in enumerate(images):
    x_start = i * (size + padding)
    # Center the text under each image
    draw.text((x_start + 10, 5), label, fill=0, font=font)

# Add arrows between images
arrow_y = label_height + size // 2
for i in range(num_images - 1):
    x_end = (i + 1) * (size + padding) - padding // 2
    draw.text((x_end - 8, arrow_y - 6), "->", fill=100, font=font)

result.save('fft_concept_diagram.png')

print("FFT concept diagram created!")
print(f"Output saved as 'fft_concept_diagram.png'")
print(f"Shows the complete FFT filtering pipeline:")
print("  Original -> FFT Spectrum -> Mask -> Filtered -> Result")
