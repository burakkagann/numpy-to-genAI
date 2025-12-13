.. _module-3-4-1-convolution:

=====================================
3.4.1 - Convolution
=====================================

:Duration: 18-20 minutes
:Level: Intermediate
:Prerequisites: Module 1.1.1 (RGB Basics), Module 2.1 (Basic Shapes)

Overview
========

In this exercise, you will learn how **convolution** transforms images by sliding a small grid of numbers (called a **kernel**) across every pixel. This fundamental operation powers everything from Instagram filters to neural network vision systems.

By the end of this module, you will understand why a simple 3x3 matrix can blur an image, detect edges, or sharpen details.

**Learning Objectives**

By completing this module, you will:

* Understand convolution as a "sliding window" operation that combines pixels with kernel weights
* Apply different kernels to achieve blur, sharpen, and edge detection effects
* Implement your own convolution function from scratch using nested loops
* Recognize convolution as the foundation of Convolutional Neural Networks (CNNs) [LeCun1998]_


Quick Start
===========

Let's see convolution in action immediately. Run this script to blur a checkerboard pattern:

.. code-block:: python
   :caption: simple_convolution.py - Blur a checkerboard pattern

   import numpy as np
   from PIL import Image

   # Create a checkerboard pattern with sharp edges
   CANVAS_SIZE = 256
   SQUARE_SIZE = 32
   canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.float64)

   for row in range(CANVAS_SIZE):
       for col in range(CANVAS_SIZE):
           square_row = row // SQUARE_SIZE
           square_col = col // SQUARE_SIZE
           if (square_row + square_col) % 2 == 0:
               canvas[row, col] = 255.0

   # Define a 5x5 blur kernel (averaging filter)
   KERNEL_SIZE = 5
   blur_kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE)) / (KERNEL_SIZE ** 2)

   # Apply convolution
   output_size = CANVAS_SIZE - KERNEL_SIZE + 1
   output = np.zeros((output_size, output_size))

   for y in range(output_size):
       for x in range(output_size):
           region = canvas[y:y + KERNEL_SIZE, x:x + KERNEL_SIZE]
           output[y, x] = np.sum(region * blur_kernel)

   result = Image.fromarray(output.astype(np.uint8), mode='L')
   result.save('simple_convolution.png')

.. figure:: simple_convolution.png
   :width: 500px
   :align: center
   :alt: Side-by-side comparison of original checkerboard and blurred result

   Left: Original sharp checkerboard pattern. Right: After convolution with a blur kernel, the sharp edges become smooth gradients.

**What just happened?** The blur kernel averaged each pixel with its neighbors, turning sharp black-white transitions into smooth gray gradients. This is the essence of convolution: transforming pixels based on their neighborhood.


Core Concepts
=============

Concept 1: What is Convolution?
-------------------------------

Convolution is a mathematical operation that combines two arrays: an **image** and a **kernel** (also called a filter). The kernel is a small matrix of weights (typically 3x3 or 5x5) that slides across the image, computing a weighted sum at each position.

**The Three Steps of Convolution:**

1. **Position** the kernel over a region of the image
2. **Multiply** each kernel value by the corresponding pixel value (element-wise)
3. **Sum** all the products to produce one output pixel

.. figure:: convolution_concept.png
   :width: 700px
   :align: center
   :alt: Diagram showing how a 3x3 kernel multiplies with image pixels and sums to produce output

   The convolution process: A 3x3 kernel overlays a region of the image. Each kernel weight is multiplied by the corresponding pixel, and the results are summed to produce a single output value.

**The Key Insight:**

The kernel values determine what the convolution *does*. A blur kernel has equal positive values (averaging neighbors). An edge detection kernel has positive values in the center and negative values around it (finding differences between neighbors).

.. tip::

   Think of convolution as asking: "What is the weighted average of this pixel's neighborhood?" The kernel weights define *how much* each neighbor contributes.


Concept 2: Understanding Kernels
--------------------------------

Different kernels produce dramatically different effects. Here are the most common types:

**Identity Kernel** (no change):

.. code-block:: python

   identity = np.array([
       [0, 0, 0],
       [0, 1, 0],
       [0, 0, 0]
   ])

The center value is 1, all others are 0. The output equals the input.

**Blur Kernel** (smoothing):

.. code-block:: python

   blur = np.array([
       [1, 1, 1],
       [1, 1, 1],
       [1, 1, 1]
   ]) / 9.0  # Normalize so values sum to 1

All values are equal, so the output is the average of the 3x3 neighborhood.

**Sharpen Kernel** (enhance edges):

.. code-block:: python

   sharpen = np.array([
       [ 0, -1,  0],
       [-1,  5, -1],
       [ 0, -1,  0]
   ])

The center value is larger than 1, neighbors are negative. This amplifies the center pixel relative to its neighbors.

**Edge Detection Kernel** (find boundaries) [Marr1980]_:

.. code-block:: python

   edge_detect = np.array([
       [-1, -1, -1],
       [-1,  8, -1],
       [-1, -1, -1]
   ])

Positive center, negative neighbors. Uniform regions become zero; boundaries become bright.

.. figure:: kernel_effects.png
   :width: 600px
   :align: center
   :alt: 2x2 grid showing original, blur, sharpen, and edge detection effects

   Different kernels produce different effects. Top-left: Original. Top-right: Blur smooths edges. Bottom-left: Sharpen enhances details. Bottom-right: Edge detection highlights boundaries.

.. important::

   **Kernel normalization matters!** For blur kernels, the values should sum to 1 to preserve brightness. For edge detection, values can sum to 0 (outputs only changes, not absolute values).

.. admonition:: Did You Know?

   The blur kernel is also called a "box filter" because it gives equal weight to all pixels in a rectangular box. Gaussian blur uses a bell-curve distribution of weights for smoother results [Gonzalez2018]_.


Hands-On Exercises
==================

Exercise 1: Execute and Explore
-------------------------------

**Time estimate:** 3-4 minutes

Run ``kernel_effects.py`` to see how different kernels transform the same image:

.. code-block:: python

   # Run this script to generate the comparison
   python kernel_effects.py

**Reflection Questions:**

1. Look at the edge detection result. Why do uniform regions (like solid shapes) appear black?
2. Why does the sharpen kernel make the image look "crisper"?
3. What would happen if you used a larger kernel (e.g., 7x7 blur)?

.. dropdown:: Solution & Explanation
   :class: note

   1. **Edge detection shows black in uniform regions** because the kernel computes differences between the center pixel and its neighbors. When all pixels are the same, the differences cancel out to zero (black).

   2. **Sharpen makes images crisper** because the negative neighbor weights subtract the "average" from the center, enhancing any difference. Edges (where values change) become more pronounced.

   3. **Larger blur kernels** produce stronger smoothing because they average over more pixels. A 7x7 blur would create a more pronounced smoothing effect than a 3x3 blur, but would also require more computation.


Exercise 2: Modify to Achieve Goals
-----------------------------------

**Time estimate:** 3-4 minutes

Starting with the blur kernel from ``simple_convolution.py``, modify the kernel values to achieve these effects:

**Goal A:** Create a horizontal edge detector (detects horizontal lines only)

.. dropdown:: Hint
   :class: tip

   Use rows with different values. Try: top row = -1, middle row = 0 or 2, bottom row = -1

.. dropdown:: Solution
   :class: note

   .. code-block:: python

      horizontal_edge = np.array([
          [-1, -1, -1],
          [ 2,  2,  2],
          [-1, -1, -1]
      ], dtype=np.float64)

   This kernel sums to 0 and computes the difference between the middle row and the rows above/below, highlighting horizontal edges.

**Goal B:** Create a vertical edge detector (detects vertical lines only)

.. dropdown:: Hint
   :class: tip

   Rotate the horizontal edge kernel by 90 degrees.

.. dropdown:: Solution
   :class: note

   .. code-block:: python

      vertical_edge = np.array([
          [-1, 2, -1],
          [-1, 2, -1],
          [-1, 2, -1]
      ], dtype=np.float64)

   This is the horizontal kernel transposed. It computes differences between the middle column and the columns on either side.

**Goal C:** Create an emboss effect (3D illusion)

.. dropdown:: Hint
   :class: tip

   Use a diagonal pattern with positive values in one corner and negative in the opposite corner.

.. dropdown:: Solution
   :class: note

   .. code-block:: python

      emboss = np.array([
          [-2, -1, 0],
          [-1,  1, 1],
          [ 0,  1, 2]
      ], dtype=np.float64)

   This creates a "lit from top-left" emboss effect by computing diagonal differences across the image.


Exercise 3: Create from Scratch
-------------------------------

**Time estimate:** 5-6 minutes

Implement your own convolution function from scratch! Complete the ``apply_convolution()`` function in the starter code to process the panda image.

**Requirements:**

* Iterate over each valid pixel position
* Extract the region under the kernel using array slicing
* Compute the element-wise product and sum
* Handle borders by padding the image

**Starter code:**

.. code-block:: python
   :caption: convolution_starter.py

   import numpy as np
   from PIL import Image

   # Load the panda image
   PANDA_PATH = '../../../3.3_artistic_filters/3.3.3_hexpanda/hexpanda/panda.png'
   panda = Image.open(PANDA_PATH).convert('L')
   panda = panda.resize((256, 256))
   image = np.array(panda, dtype=np.float64)

   # Edge detection kernel
   kernel = np.array([
       [-1, -1, -1],
       [-1,  8, -1],
       [-1, -1, -1]
   ], dtype=np.float64)

   def apply_convolution(image, kernel):
       """Apply a convolution kernel to a grayscale image."""
       kernel_size = kernel.shape[0]
       pad = kernel_size // 2

       height, width = image.shape
       output = np.zeros((height, width), dtype=np.float64)

       # Pad the image to handle borders
       padded = np.pad(image, pad, mode='edge')

       # TODO: Implement the convolution!
       # for y in range(...):
       #     for x in range(...):
       #         region = ...
       #         output[y, x] = ...

       return output

.. dropdown:: Hint 1: Loop bounds
   :class: tip

   Loop from 0 to ``height`` and 0 to ``width``. Because we padded the image, we can safely access ``padded[y:y+kernel_size, x:x+kernel_size]`` for all valid positions.

.. dropdown:: Hint 2: Extracting the region
   :class: tip

   Use NumPy slicing: ``region = padded[y:y + kernel_size, x:x + kernel_size]``

.. dropdown:: Complete Solution
   :class: note

   .. code-block:: python
      :linenos:
      :emphasize-lines: 19-24

      def apply_convolution(image, kernel):
          """Apply a convolution kernel to a grayscale image."""
          kernel_size = kernel.shape[0]
          pad = kernel_size // 2

          height, width = image.shape
          output = np.zeros((height, width), dtype=np.float64)

          # Pad the image to handle borders
          padded = np.pad(image, pad, mode='edge')

          # The core convolution loop
          for y in range(height):
              for x in range(width):
                  # Extract the region under the kernel
                  region = padded[y:y + kernel_size, x:x + kernel_size]

                  # Element-wise multiply and sum
                  output[y, x] = np.sum(region * kernel)

          return output

   The key insight is that padding allows us to process every pixel, including edge pixels, without special boundary handling code.

.. figure:: convolution_comparison.png
   :width: 600px
   :align: center
   :alt: Side-by-side comparison of original panda and edge-detected result

   Your convolution result: The panda's edges are now clearly visible as bright lines on a dark background.

**Challenge Extension:**

Modify your function to handle color images by applying convolution to each RGB channel separately, then combining the results.


Summary
=======

In this module, you learned the fundamental operation behind image filters and neural network vision systems.

**Key Takeaways:**

* Convolution slides a **kernel** across an image, computing weighted sums at each position
* The **kernel values** determine the effect: blur (averaging), sharpen (center emphasis), edge detection (difference detection)
* **Normalization** matters: blur kernels should sum to 1 to preserve brightness
* This operation is the foundation of **Convolutional Neural Networks** (CNNs)

**Common Pitfalls to Avoid:**

* Forgetting to normalize blur kernels (causes image brightening or darkening)
* Not handling borders correctly (causes output size reduction or edge artifacts)
* Using integer math when float precision is needed (causes rounding errors)

**Why This Matters:**

Convolution is everywhere in modern computer vision. When you apply a filter in Photoshop, your phone detects faces, or an AI classifies images, convolution is doing the heavy lifting. Understanding this operation gives you insight into how machines "see."


Further Exploration
===================

For a more interactive exploration of convolution concepts, see the Jupyter notebook:

* ``GenerativeConvolution.ipynb`` - Interactive examples with parameter sliders

For built-in convolution filters, see the Pillow ImageFilter module [PILDocs]_.

References
==========

.. [Gonzalez2018] Gonzalez, R.C. and Woods, R.E. (2018). *Digital Image Processing* (4th ed.). Pearson. Chapter 3: Intensity Transformations and Spatial Filtering. [Foundational textbook on image processing including convolution theory]

.. [Sobel1968] Sobel, I. (1968). "A 3x3 Isotropic Gradient Operator for Image Processing." Presented at the Stanford Artificial Intelligence Project. [Original edge detection operator using convolution]

.. [LeCun1998] LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. (1998). "Gradient-based learning applied to document recognition." *Proceedings of the IEEE*, 86(11), 2278-2324. [Seminal paper on CNNs using convolution for feature learning]

.. [Marr1980] Marr, D. and Hildreth, E. (1980). "Theory of edge detection." *Proceedings of the Royal Society of London B*, 207(1167), 187-217. [Theoretical foundations of edge detection]

.. [NumPyDocs] Harris, C.R., et al. (2020). "Array programming with NumPy." *Nature*, 585, 357-362. https://doi.org/10.1038/s41586-020-2649-2 [NumPy array operations used in convolution]

.. [PILDocs] Clark, A., et al. (2024). *Pillow (PIL Fork) Documentation*. https://pillow.readthedocs.io/ [ImageFilter module for built-in convolution kernels]

.. [Prewitt1970] Prewitt, J.M.S. (1970). "Object enhancement and extraction." *Picture Processing and Psychopictorics*, 10(1), 15-19. [Alternative edge detection operator]
