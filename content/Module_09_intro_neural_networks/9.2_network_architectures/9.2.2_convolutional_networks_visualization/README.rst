.. _module-9-2-2-convolutional-networks-visualization:

=============================================
9.2.2 Convolutional Networks Visualization
=============================================

:Duration: 35-40 minutes
:Level: Intermediate

Overview
========

Convolutional Neural Networks (CNNs) revolutionized computer vision by learning to detect visual features automatically. Unlike fully-connected networks that treat images as flat vectors, CNNs preserve spatial relationships and learn hierarchical patterns, from simple edges to complex textures to high-level concepts. In this exercise, you will build a 2D convolution operation from scratch and visualize how filters detect features in images.

The key insight behind CNNs is that the same pattern can appear anywhere in an image. A vertical edge is a vertical edge whether it appears in the top-left corner or bottom-right. CNNs exploit this by sliding small filters (kernels) across the entire image, applying the same weights everywhere. This weight sharing dramatically reduces the number of parameters compared to fully-connected networks while making CNNs translation-invariant [LeCun1998]_.

This exercise focuses on the fundamental convolution operation and filter visualization. Understanding what happens inside a single convolutional layer provides the foundation for interpreting more complex architectures and working with generative models in later modules.

Learning Objectives
-------------------

By the end of this exercise, you will be able to:

* Understand how 2D convolution works as a sliding window operation
* Implement convolution from scratch using NumPy loops
* Recognize how different filter kernels detect different features (edges, blur, sharpen)
* Create visualizations showing filter activations as feature maps


Quick Start: See It In Action
=============================

Run this code to apply Sobel edge detection using 2D convolution:

.. code-block:: python
   :caption: Detect edges using convolution
   :linenos:

   import numpy as np
   from PIL import Image

   def convolve2d(image, kernel):
       h, w = image.shape
       kh, kw = kernel.shape
       output = np.zeros((h - kh + 1, w - kw + 1))
       for y in range(output.shape[0]):
           for x in range(output.shape[1]):
               region = image[y:y+kh, x:x+kw]
               output[y, x] = np.sum(region * kernel)
       return output

   # Create a simple test pattern with shapes
   pattern = np.zeros((256, 256))
   pattern[30:90, 30:110] = 255  # Rectangle
   pattern[150:220, 50:100] = 255  # Another shape

   # Sobel X kernel detects vertical edges
   sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)

   # Apply convolution
   edges = convolve2d(pattern, sobel_x)
   edges = np.clip(edges, 0, 255).astype(np.uint8)

   Image.fromarray(edges).save("edges_output.png")
   print("Edge detection complete!")

.. figure:: edge_detection_output.png
   :width: 532px
   :align: center
   :alt: Side-by-side comparison showing original geometric shapes on left and detected vertical edges on right

   Original shapes (left) and Sobel edge detection output (right). The filter highlights vertical edges where pixel intensity changes rapidly from dark to light or light to dark.

The convolution operation slides a small filter across the image, computing weighted sums at each position. This simple operation, repeated with different learned filters, forms the foundation of how CNNs extract features from images.


Core Concepts
=============

Concept 1: What is a Convolutional Neural Network?
--------------------------------------------------

A **Convolutional Neural Network (CNN)** is a type of neural network designed specifically for processing grid-like data such as images. Instead of connecting every input pixel to every hidden neuron, CNNs use local connectivity and shared weights to efficiently learn spatial hierarchies of features [Goodfellow2016]_.

.. figure:: cnn_architecture.png
   :width: 700px
   :align: center
   :alt: CNN architecture diagram showing progression from input image through convolutional layers, pooling layers, and fully connected layers to output

   A typical CNN architecture. Input images pass through convolutional layers that extract features, pooling layers that reduce spatial dimensions, and finally fully connected layers for classification.

The key components of a CNN are:

* **Convolutional layers**: Apply learnable filters to detect local patterns
* **Pooling layers**: Reduce spatial dimensions while preserving important features
* **Fully connected layers**: Combine features for final predictions

The power of CNNs comes from their ability to learn what filters to use during training. Early layers typically learn simple features like edges and textures, while deeper layers learn complex patterns like shapes and object parts [Zeiler2014]_.

.. admonition:: Did You Know?

   The 2012 AlexNet architecture used CNNs to win the ImageNet competition by a huge margin, reducing the top-5 error rate from ~26% to ~16% [Krizhevsky2012]_. This breakthrough triggered the deep learning revolution and demonstrated that CNNs could learn powerful visual representations when trained on large datasets with GPUs.


Concept 2: The Convolution Operation
------------------------------------

**Convolution** is the fundamental operation in CNNs. It involves sliding a small filter (kernel) across an image and computing weighted sums at each position [Gonzalez2018]_. The filter values act as weights that determine what pattern the filter responds to.

.. figure:: convolution_animation.gif
   :width: 700px
   :align: center
   :alt: Animation showing a 3x3 kernel sliding across an 8x8 input grid, with element-wise multiplication and summation displayed at each position

   The convolution operation in action. A 3x3 kernel slides across the input, computing element-wise products and summing them to produce each output value.

Here is a pure NumPy implementation of 2D convolution:

.. code-block:: python
   :caption: 2D convolution implementation
   :linenos:
   :emphasize-lines: 12,14,15

   def convolve2d(image, kernel):
       """Perform 2D convolution on a grayscale image."""
       image_height, image_width = image.shape
       kernel_height, kernel_width = kernel.shape

       # Output is smaller due to valid convolution (no padding)
       output_height = image_height - kernel_height + 1
       output_width = image_width - kernel_width + 1
       output = np.zeros((output_height, output_width))

       # Slide the kernel across the image
       for y in range(output_height):
           for x in range(output_width):
               region = image[y:y+kernel_height, x:x+kernel_width]
               output[y, x] = np.sum(region * kernel)

       return output

**Line 12-13**: The nested loops slide the kernel to every valid position in the image.

**Line 14**: Extracts the region of the image currently under the kernel.

**Line 15**: Computes the element-wise product of the region and kernel, then sums all values to produce a single output pixel. This operation uses NumPy's array slicing and element-wise multiplication [NumPyDocs922]_.

The output image is smaller than the input because we only place the kernel where it fully overlaps with the image (valid convolution). For a 3x3 kernel, the output loses 2 pixels from each dimension.


Concept 3: Filter Kernels and Feature Detection
-----------------------------------------------

Different filter kernels detect different features. Classical image processing provides several well-known kernels that we can use to understand what CNNs learn [Szeliski2022]_.

.. figure:: filter_kernels.png
   :width: 550px
   :align: center
   :alt: Grid showing six common filter kernels with their values displayed: Sobel X, Sobel Y, Gaussian, Sharpen, Emboss, and Laplacian

   Common filter kernels. Green values are positive (increase output), red values are negative (decrease output), and white values are zero.

**Edge Detection Filters (Sobel)**:

.. code-block:: python
   :caption: Sobel filters for edge detection

   # Detects vertical edges (horizontal gradients)
   sobel_x = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])

   # Detects horizontal edges (vertical gradients)
   sobel_y = np.array([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]])

The Sobel X filter responds strongly where pixel values change from left to right (vertical edges). Sobel Y responds to changes from top to bottom (horizontal edges).

**Blur and Sharpen Filters**:

.. code-block:: python
   :caption: Gaussian blur and sharpen filters

   # Gaussian blur (smoothing)
   gaussian = np.array([[1, 2, 1],
                        [2, 4, 2],
                        [1, 2, 1]]) / 16.0

   # Sharpen (enhance edges)
   sharpen = np.array([[ 0, -1,  0],
                       [-1,  5, -1],
                       [ 0, -1,  0]])

Gaussian blur averages neighboring pixels, creating a smoothing effect. The sharpen filter does the opposite, enhancing edges by subtracting the surrounding average from the center pixel.

.. figure:: feature_maps_grid.png
   :width: 550px
   :align: center
   :alt: Grid showing the same input image processed by six different filters, demonstrating how each kernel produces a different feature map

   The same input processed by different filters. Each filter produces a feature map highlighting different aspects of the image.

When training a CNN, the network learns optimal filter values through backpropagation. Instead of using hand-crafted Sobel or Gaussian filters, the CNN discovers filters that best solve the specific task at hand. Recent research on feature visualization has revealed surprising insights into what filters learn at different layers [Olah2017]_.


Hands-On Exercises
==================

Exercise 1: Execute and Explore
-------------------------------

Run the complete convolution implementation to see edge detection in action:

.. code-block:: python
   :caption: cnn_visualization.py
   :linenos:

   import numpy as np
   from PIL import Image

   def convolve2d(image, kernel):
       h, w = image.shape
       kh, kw = kernel.shape
       output_h = h - kh + 1
       output_w = w - kw + 1
       output = np.zeros((output_h, output_w), dtype=np.float64)

       for y in range(output_h):
           for x in range(output_w):
               region = image[y:y+kh, x:x+kw]
               output[y, x] = np.sum(region * kernel)
       return output

   def normalize(image):
       if image.max() == image.min():
           return np.zeros_like(image, dtype=np.uint8)
       return ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

   # Create geometric shapes test pattern
   pattern = np.zeros((256, 256), dtype=np.float64)
   pattern[30:90, 30:110] = 255  # Rectangle
   center_y, center_x = 60, 190
   y, x = np.ogrid[:256, :256]
   mask = (x - center_x)**2 + (y - center_y)**2 <= 40**2
   pattern[mask] = 255  # Circle

   # Define Sobel filters
   sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
   sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

   # Apply both edge detectors
   edges_x = convolve2d(pattern, sobel_x)
   edges_y = convolve2d(pattern, sobel_y)

   # Combine into edge magnitude
   magnitude = np.sqrt(edges_x**2 + edges_y**2)
   result = normalize(magnitude)

   Image.fromarray(result).save("edges_combined.png")
   print(f"Input shape: {pattern.shape}")
   print(f"Output shape: {result.shape}")
   print(f"Edge magnitude range: [{magnitude.min():.1f}, {magnitude.max():.1f}]")

After running the code, answer these reflection questions:

1. Why is the output image smaller than the input image?
2. What is the difference between applying Sobel X versus Sobel Y to an image?
3. Why do we combine edges_x and edges_y using the square root of the sum of squares?
4. What would happen if all kernel values were positive?

.. dropdown:: Answers and Explanation
   :class-title: sd-font-weight-bold

   1. **Smaller output**: The output is smaller because we use valid convolution (no padding). With a 3x3 kernel on a 256x256 image, the output is 254x254. The kernel cannot be centered on edge pixels because it would extend outside the image.

   2. **Sobel X vs Y**: Sobel X detects vertical edges (changes in the horizontal direction) while Sobel Y detects horizontal edges (changes in the vertical direction). Applied to a rectangle, Sobel X highlights the left and right sides while Sobel Y highlights the top and bottom.

   3. **Magnitude formula**: The combined magnitude sqrt(Gx^2 + Gy^2) gives the total edge strength regardless of direction. This is the Euclidean norm of the gradient vector at each pixel. Using abs(Gx) + abs(Gy) would also work but is less accurate.

   4. **All positive values**: A kernel with all positive values acts as a blur filter. It averages the local neighborhood, smoothing the image. Edge detection requires both positive and negative values to compute differences (gradients).


Exercise 2: Modify Parameters
-----------------------------

Experiment with different filter kernels to understand their effects.

**Goal 1**: Compare horizontal and vertical edge detection

Apply Sobel X and Sobel Y separately to see which edges each filter detects:

.. code-block:: python
   :caption: Compare edge directions

   # Vertical edge detector (responds to horizontal gradients)
   sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)

   # Horizontal edge detector (responds to vertical gradients)
   sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

   edges_x = normalize(convolve2d(pattern, sobel_x))
   edges_y = normalize(convolve2d(pattern, sobel_y))

   Image.fromarray(edges_x).save("edges_vertical.png")
   Image.fromarray(edges_y).save("edges_horizontal.png")

.. dropdown:: Hint: Understanding edge directions
   :class-title: sd-font-weight-bold

   The naming can be confusing. Sobel X computes the gradient in the X direction (horizontal), which means it responds to vertical edges. Think of it as detecting how much the image changes as you move from left to right.

**Goal 2**: Create artistic filter effects

Combine multiple filters for artistic results:

.. code-block:: python
   :caption: Artistic filter combination

   # Apply sharpen first, then edge detection
   sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float64)

   sharpened = convolve2d(pattern, sharpen)
   edges = convolve2d(normalize(sharpened).astype(np.float64), sobel_x)

   # Create colorful output
   result_rgb = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
   normalized = normalize(edges)
   result_rgb[:, :, 0] = normalized  # Red channel
   result_rgb[:, :, 1] = 255 - normalized  # Inverted green
   result_rgb[:, :, 2] = 128  # Constant blue

   Image.fromarray(result_rgb).save("artistic_edges.png")

.. dropdown:: Hint: Chaining filters
   :class-title: sd-font-weight-bold

   When chaining filters, remember that each convolution shrinks the image. The output of the first filter becomes the input to the second. Make sure to normalize between steps if the intermediate values are outside 0-255.

**Goal 3**: Experiment with kernel size

Try a larger 5x5 Gaussian blur kernel and compare to 3x3:

.. code-block:: python
   :caption: Compare kernel sizes

   # 3x3 Gaussian
   gaussian_3x3 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0

   # 5x5 Gaussian (more blur)
   gaussian_5x5 = np.array([
       [1,  4,  6,  4, 1],
       [4, 16, 24, 16, 4],
       [6, 24, 36, 24, 6],
       [4, 16, 24, 16, 4],
       [1,  4,  6,  4, 1]
   ]) / 256.0

   blur_3x3 = normalize(convolve2d(pattern, gaussian_3x3))
   blur_5x5 = normalize(convolve2d(pattern, gaussian_5x5))

.. dropdown:: Hint: Kernel size effects
   :class-title: sd-font-weight-bold

   Larger kernels affect more neighboring pixels, producing stronger effects. A 5x5 blur is more pronounced than 3x3. However, larger kernels are computationally more expensive and shrink the output more.


Exercise 3: Create Your Own
---------------------------

Implement convolution from scratch and create a custom artistic filter.

**Requirements**:

* Complete the ``convolve2d`` function with correct loop bounds
* Create a custom 3x3 kernel that produces an interesting visual effect
* Apply your kernel to the test pattern and save the result

**Starter Code**:

.. code-block:: python
   :caption: cnn_starter.py (complete the TODO sections)
   :linenos:

   import numpy as np
   from PIL import Image

   def convolve2d(image, kernel):
       image_height, image_width = image.shape
       kernel_height, kernel_width = kernel.shape

       # TODO: Calculate output dimensions
       output_height = None  # Replace
       output_width = None   # Replace

       # TODO: Initialize output array
       output = None  # Replace

       # TODO: Implement convolution loops
       for y in range(output_height):
           for x in range(output_width):
               # TODO: Extract region and compute output
               pass

       return output

   # Create test pattern
   pattern = np.zeros((256, 256), dtype=np.float64)
   for y in range(256):
       for x in range(256):
           if ((x // 32) + (y // 32)) % 2 == 0:
               pattern[y, x] = 255.0

   # TODO: Define your custom kernel
   my_kernel = np.array([
       [0, 0, 0],
       [0, 0, 0],
       [0, 0, 0]
   ], dtype=np.float64)

   # Apply and save
   result = convolve2d(pattern, my_kernel)

.. dropdown:: Hint 1: Output dimensions
   :class-title: sd-font-weight-bold

   For valid convolution (no padding), the output dimensions are:

   .. code-block:: python

      output_height = image_height - kernel_height + 1
      output_width = image_width - kernel_width + 1

   With a 3x3 kernel on a 256x256 image, you get a 254x254 output.

.. dropdown:: Hint 2: The convolution loop
   :class-title: sd-font-weight-bold

   Inside the loop, extract a region of the same size as the kernel and compute the sum of element-wise products:

   .. code-block:: python

      for y in range(output_height):
          for x in range(output_width):
              region = image[y:y+kernel_height, x:x+kernel_width]
              output[y, x] = np.sum(region * kernel)

.. dropdown:: Hint 3: Custom kernel ideas
   :class-title: sd-font-weight-bold

   Some interesting kernels to try:

   .. code-block:: python

      # Diagonal edge detector
      diagonal = np.array([[-1, 0, 1],
                           [ 0, 0, 0],
                           [ 1, 0, -1]])

      # Emboss (3D shadow effect)
      emboss = np.array([[-2, -1, 0],
                         [-1,  1, 1],
                         [ 0,  1, 2]])

      # High-pass (inverse of blur)
      high_pass = np.array([[-1, -1, -1],
                            [-1,  8, -1],
                            [-1, -1, -1]])

.. dropdown:: Complete Solution
   :class-title: sd-font-weight-bold

   .. code-block:: python
      :linenos:

      import numpy as np
      from PIL import Image

      def convolve2d(image, kernel):
          image_height, image_width = image.shape
          kernel_height, kernel_width = kernel.shape

          output_height = image_height - kernel_height + 1
          output_width = image_width - kernel_width + 1
          output = np.zeros((output_height, output_width), dtype=np.float64)

          for y in range(output_height):
              for x in range(output_width):
                  region = image[y:y+kernel_height, x:x+kernel_width]
                  output[y, x] = np.sum(region * kernel)

          return output

      def normalize(image):
          if image.max() == image.min():
              return np.zeros_like(image, dtype=np.uint8)
          return ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

      # Create checkerboard pattern
      pattern = np.zeros((256, 256), dtype=np.float64)
      for y in range(256):
          for x in range(256):
              if ((x // 32) + (y // 32)) % 2 == 0:
                  pattern[y, x] = 255.0

      # Custom emboss kernel
      emboss = np.array([[-2, -1, 0],
                         [-1,  1, 1],
                         [ 0,  1, 2]], dtype=np.float64)

      result = convolve2d(pattern, emboss)
      output = normalize(result)
      Image.fromarray(output).save("my_filter_output.png")
      print(f"Output saved! Shape: {output.shape}")

**Challenge Extension**: Create an animated GIF that shows a filter being applied with varying strength, smoothly transitioning from no effect (identity) to full effect.

.. dropdown:: Challenge Solution
   :class-title: sd-font-weight-bold

   .. code-block:: python

      import imageio.v2 as imageio

      # Identity kernel (no change)
      identity = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float64)

      # Target effect (emboss)
      emboss = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float64)

      frames = []
      for t in np.linspace(0, 1, 20):
          # Interpolate between identity and emboss
          kernel = (1 - t) * identity + t * emboss
          result = convolve2d(pattern, kernel)
          frames.append(normalize(result))

      imageio.mimsave("filter_transition.gif", frames, fps=10)

.. figure:: artistic_filters.png
   :width: 400px
   :align: center
   :alt: Artistic filter output showing colorful edge-enhanced pattern

   Example artistic output combining multiple filters with color mapping.


Summary
=======

Key Takeaways
-------------

* **Convolution** slides a small filter (kernel) across an image, computing weighted sums at each position
* The **output size** shrinks because the kernel cannot be centered on edge pixels (valid convolution)
* **Sobel filters** detect edges: Sobel X finds vertical edges, Sobel Y finds horizontal edges
* **Edge magnitude** combines directional edges using sqrt(Gx^2 + Gy^2)
* CNNs **learn** optimal filter values during training rather than using hand-crafted filters
* Multiple convolutional layers create **feature hierarchies**, from simple edges to complex patterns

Common Pitfalls
---------------

* **Index out of bounds**: The output is smaller than input; account for kernel size in your loops
* **Integer overflow**: Use float64 for intermediate calculations, then normalize to uint8 for display
* **Forgetting normalization**: Convolution outputs may have negative values or exceed 255
* **Wrong kernel orientation**: NumPy uses [row, col] indexing, which maps to [y, x]
* **Confusing edge direction**: Sobel X detects vertical edges (it measures horizontal gradient)
* **Image format issues**: Always convert arrays to uint8 before saving with Pillow [PillowDocs922]_



References
==========

.. [LeCun1998] LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278-2324. https://doi.org/10.1109/5.726791

.. [Krizhevsky2012] Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. *Advances in Neural Information Processing Systems*, 25, 1097-1105. https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html

.. [Goodfellow2016] Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. ISBN: 978-0-262-03561-3. https://www.deeplearningbook.org/

.. [Gonzalez2018] Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing* (4th ed.). Pearson. ISBN: 978-0-13-335672-4

.. [Zeiler2014] Zeiler, M. D., & Fergus, R. (2014). Visualizing and understanding convolutional networks. In *European Conference on Computer Vision* (pp. 818-833). Springer. https://arxiv.org/abs/1311.2901

.. [Olah2017] Olah, C., Mordvintsev, A., & Schubert, L. (2017). Feature visualization. *Distill*. https://distill.pub/2017/feature-visualization/

.. [Szeliski2022] Szeliski, R. (2022). *Computer Vision: Algorithms and Applications* (2nd ed.). Springer. https://szeliski.org/Book/

.. [NumPyDocs922] NumPy Developers. (2024). NumPy array slicing. *NumPy Documentation*. https://numpy.org/doc/stable/user/basics.indexing.html

.. [PillowDocs922] Clark, A., et al. (2024). *Pillow: Python Imaging Library* (Version 10.2.0). Python Software Foundation. https://pillow.readthedocs.io/

.. [Sobel1968] Sobel, I., & Feldman, G. (1968). A 3x3 isotropic gradient operator for image processing. Unpublished, presented at the Stanford Artificial Intelligence Laboratory (SAIL).
