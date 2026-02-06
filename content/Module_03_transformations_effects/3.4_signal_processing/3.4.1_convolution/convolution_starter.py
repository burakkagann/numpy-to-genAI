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

Implementation inspired by:
- SciPy ndimage.convolve documentation
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.convolve.html
- Gonzalez, R.C. and Woods, R.E. (2018). Digital Image Processing, 4th ed.
  Chapter 3: Intensity Transformations and Spatial Filtering
"""

import numpy as np
from PIL import Image
import os

# =============================================================================
# Step 1: Load the Brandenburg Gate image for testing
# =============================================================================
# Path to the shared Brandenburg Gate image (relative from this directory)
IMAGE_PATH = '../../../../../_static/images/bbtor.jpg'

# Check if the image file exists
if os.path.exists(IMAGE_PATH):
    # Load image and convert to grayscale
    img = Image.open(IMAGE_PATH).convert('L')

    # Resize for faster processing
    img = img.resize((256, 171))

    # Convert to numpy array
    image = np.array(img, dtype=np.float64)
    print(f"Loaded Brandenburg Gate image: {image.shape[1]}x{image.shape[0]} pixels")
else:
    print(f"Warning: Image not found at {IMAGE_PATH}")
    print("Please ensure the image exists at the specified path.")
    # Create simple fallback
    image = np.zeros((171, 256), dtype=np.float64)
    image[50:120, 80:180] = 255  # White rectangle

# =============================================================================
# Step 2: Define a kernel to test with
# =============================================================================
# Try different kernels! Here's an edge detection kernel (Laplacian):
# - Positive center (8) surrounded by negative neighbors (-1 each)
# - Sum of weights = 0, so uniform regions become zero
kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=np.float64)

print("\nKernel to apply:")
print(kernel)

# =============================================================================
# Step 3: TODO - Complete this function!
# =============================================================================
def apply_convolution(image, kernel):
    """
    Apply a convolution kernel to a grayscale image.

    The convolution formula (from SciPy documentation):
    C_i = Σ_j{I_{i+k-j} W_j}

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
    # 'edge' mode repeats the outermost pixels outward
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
# Step 4: Test your implementation
# =============================================================================
if __name__ == '__main__':
    print("\nApplying your convolution function...")

    # Apply your convolution function
    result = apply_convolution(image, kernel)

    # Clip to valid range and convert to image
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Save result
    output_image = Image.fromarray(result, mode='L')
    output_image.save('my_convolution_result.png')

    print("Your convolution result saved as my_convolution_result.png")
    print("If the output looks like edge detection, you did it right!")
    print("\nHint: If the output is all black, your convolution loop isn't implemented yet.")
