
import numpy as np
from PIL import Image, ImageDraw

# Canvas setup
width, height = 512, 512
center_x, center_y = width // 2, height // 2

# Create RGB image with dark background
image = np.zeros((height, width, 3), dtype=np.uint8)
image[:, :] = [20, 20, 30]  # Dark blue-gray background

# Convert to PIL for drawing
pil_image = Image.fromarray(image, mode='RGB')
draw = ImageDraw.Draw(pil_image)

# Draw concentric circles (radius guides)
for radius in range(50, 200, 50):
    draw.ellipse(
        [center_x - radius, center_y - radius,
         center_x + radius, center_y + radius],
        outline=(60, 60, 80), width=1
    )

# Draw axes
draw.line([(50, center_y), (width - 50, center_y)], fill=(100, 100, 120), width=1)
draw.line([(center_x, 50), (center_x, height - 50)], fill=(100, 100, 120), width=1)

# Example point in polar coordinates
angle_degrees = 45
radius = 150
angle_radians = np.radians(angle_degrees)

# Convert polar to Cartesian
point_x = center_x + int(radius * np.cos(angle_radians))
point_y = center_y - int(radius * np.sin(angle_radians))  # Subtract for screen coords

# Draw radius line from center to point
draw.line([(center_x, center_y), (point_x, point_y)], fill=(255, 150, 50), width=3)

# Draw angle arc
arc_radius = 50
draw.arc(
    [center_x - arc_radius, center_y - arc_radius,
     center_x + arc_radius, center_y + arc_radius],
    start=-angle_degrees, end=0, fill=(100, 200, 255), width=2
)

# Draw the point
point_radius = 8
draw.ellipse(
    [point_x - point_radius, point_y - point_radius,
     point_x + point_radius, point_y + point_radius],
    fill=(255, 100, 100)
)

# Draw projection lines (dashed effect with short segments)
# Vertical projection
for y in range(center_y, point_y, -8):
    draw.line([(point_x, y), (point_x, max(y - 4, point_y))], fill=(150, 150, 150), width=1)
# Horizontal projection
for x in range(center_x, point_x, 8):
    draw.line([(x, point_y), (min(x + 4, point_x), point_y)], fill=(150, 150, 150), width=1)

# Add labels using simple text (positions approximate)
# r label - along radius line
label_r_x = center_x + int(75 * np.cos(angle_radians)) - 15
label_r_y = center_y - int(75 * np.sin(angle_radians)) - 15
draw.text((label_r_x, label_r_y), "r", fill=(255, 150, 50))

# theta label - near arc
draw.text((center_x + 55, center_y - 25), "theta", fill=(100, 200, 255))

# Point label
draw.text((point_x + 12, point_y - 8), "(x, y)", fill=(255, 100, 100))

# Origin label
draw.text((center_x + 5, center_y + 5), "O", fill=(200, 200, 200))

# Axis labels
draw.text((width - 45, center_y + 5), "x", fill=(150, 150, 150))
draw.text((center_x + 5, 55), "y", fill=(150, 150, 150))

# Convert back to numpy and save
result = np.array(pil_image)
output_image = Image.fromarray(result, mode='RGB')
output_image.save('polar_coordinate_diagram.png')
