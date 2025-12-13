"""
Exercise 3: Implement Your Own Convolution - Starter Code

Your task: Complete the apply_convolution() function to implement
2D convolution from scratch using nested loops.

Requirements:
1. Iterate over each pixel position in the image
2. Extract the region under the kernel
3. Multiply region by kernel (element-wise)
4. Sum the result to get the output pixel value
5. Handle borders by using edge padding

Author: Your Name
Date: Today's Date
"""

import numpy as np
from PIL import Image

# =============================================================================
# Load the panda image for testing
# =============================================================================
# Path to shared panda image (relative from this directory)
PANDA_PATH = '../../../3.3_artistic_filters/3.3.3_hexpanda/hexpanda/panda.png'

# Load and convert to grayscale
panda = Image.open(PANDA_PATH).convert('L')
panda = panda.resize((256, 256))  # Resize for faster processing
image = np.array(panda, dtype=np.float64)

# =============================================================================
# Define a kernel to test with
# =============================================================================
# Try different kernels! Here's an edge detection kernel:
kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=np.float64)

# =============================================================================
# TODO: Complete this function!
# =============================================================================
def apply_convolution(image, kernel):
    """
    Apply a convolution kernel to a grayscale image.

    Parameters:
        image (np.ndarray): 2D array of pixel values (grayscale)
        kernel (np.ndarray): 2D array of kernel weights (e.g., 3x3)

    Returns:
        np.ndarray: Convolved image (same size as input)
    """
    kernel_size = kernel.shape[0]
    pad = kernel_size // 2

    # Create output array (same size as input)
    height, width = image.shape
    output = np.zeros((height, width), dtype=np.float64)

    # Pad the image to handle borders
    padded = np.pad(image, pad, mode='edge')

    # TODO: Implement the convolution!
    # Hint 1: Use nested for loops to iterate over each pixel
    # Hint 2: Extract a region from 'padded' using slicing
    # Hint 3: Multiply the region by the kernel and sum

    # YOUR CODE HERE:
    # for y in range(...):
    #     for x in range(...):
    #         region = ...
    #         output[y, x] = ...

    return output


# =============================================================================
# Test your implementation
# =============================================================================
if __name__ == '__main__':
    # Apply your convolution function
    result = apply_convolution(image, kernel)

    # Clip to valid range and convert to image
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Save result
    output_image = Image.fromarray(result, mode='L')
    output_image.save('my_convolution_result.png')

    print("Your convolution result saved as my_convolution_result.png")
    print("If the output looks like edge detection, you did it right!")
