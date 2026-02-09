# Training Images

This directory should contain 10 African fabric images for DreamBooth/LoRA training.

## Expected Contents

Copy 10 diverse fabric images from the DCGAN dataset:

```
content/Module_12_generative_ai_models/12.1_generative_adversarial_networks/12.1.2_dcgan_art/african_fabric_dataset/
```

Name them as:
- fabric_001.png
- fabric_002.png
- ...
- fabric_010.png

## Selection Guidelines

Choose images with:
- **Varied colors**: Mix vibrant reds, blues, yellows, earth tones
- **Different patterns**: Geometric, floral, abstract, striped
- **Various orientations**: Horizontal, vertical, diagonal designs
- **Consistent quality**: Clear, well-lit, minimal distortions

## Image Requirements

- Resolution: 512x512 pixels (will be resized if needed)
- Format: PNG or JPEG
- Color mode: RGB
- Quantity: 10 images recommended

## Why 10 Images?

- Too few (< 5): May not capture pattern diversity
- Too many (> 20): Longer training without significant improvement
- 10 images: Good balance for learning style while maintaining diversity
