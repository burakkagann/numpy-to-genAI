import numpy as np
from PIL import Image

array = np.zeros((200, 200), dtype=np.uint8)
array += 255  # Add 255 to every element

image = Image.fromarray(array)
image.save('white_square.png')