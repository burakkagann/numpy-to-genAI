from PIL import Image
from noise import pnoise2

# Image settings
width, height = 512, 512
scale = 100.0  # Controls zoom level

# Create image
img = Image.new('RGB', (width, height))
pixels = img.load()

# Generate Perlin noise
for y in range(height):
    for x in range(width):
        # Get Perlin noise value (-1 to 1)
        noise_val = pnoise2(x / scale, y / scale, octaves=6)

        # Map to 0-255 range
        color = int((noise_val + 1) * 127.5)

        # Create cloud-like blue tones
        pixels[x, y] = (color, color, 255)

img.save('perlin_clouds.png')
img.show()