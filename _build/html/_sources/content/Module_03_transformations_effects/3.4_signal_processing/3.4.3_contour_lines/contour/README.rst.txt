.. _module-3-4-3-contour-lines:

=====================================
3.4.3 - Contour Lines
=====================================

:Duration: 15-18 minutes
:Level: Beginner-Intermediate
:Prerequisites: Module 1.1.1 (Images as Arrays), Module 3.4.1 (Convolution basics)

Overview
========

Contour lines are one of the most intuitive ways to visualize continuous data on a 2D surface. From topographic maps showing mountain elevations to weather maps displaying atmospheric pressure, contours transform complex scalar fields into easily readable patterns. In this module, you will create contour visualizations from synthetic terrain data using NumPy arrays.

**Learning Objectives**

By completing this module, you will:

* Understand scalar fields as 2D arrays representing height or intensity
* Create stepped contour bands using integer quantization
* Generate thin isolines using modulo operations
* Apply Gaussian functions to create terrain-like data

Quick Start: Your First Contour Map
====================================

Let's start with something visual. Run this code to create a simple contour visualization:

.. code-block:: python
   :caption: Create a stepped contour from a Gaussian hill
   :linenos:

   import numpy as np
   from PIL import Image

   # Step 1: Create a coordinate grid (like a map)
   size = 300
   x = np.linspace(-5, 5, size)
   y = np.linspace(-5, 5, size)
   X, Y = np.meshgrid(x, y)

   # Step 2: Create a single Gaussian "hill" centered at origin
   height = np.exp(-(X**2 + Y**2) / 4)

   # Step 3: Normalize to 0-1 range and quantize into 8 levels
   normalized = (height - height.min()) / (height.max() - height.min())
   contour = (normalized * 8).astype(np.uint8) * 32

   # Step 4: Save the result
   image = Image.fromarray(contour, mode='L')
   image.save('simple_contour.png')
   print("Done! Saved simple_contour.png")

.. figure:: simple_contour.png
   :width: 300px
   :align: center
   :alt: Stepped contour visualization showing concentric rings from dark center to light edges

   A single Gaussian hill visualized as stepped contour bands

.. tip::

   Notice how the continuous "hill" becomes discrete bands. Each gray level represents a range of heights, just like elevation colors on a topographic map.

Understanding Contour Lines
============================

The scalar field concept
------------------------

A **scalar field** is simply a 2D array where each cell contains a single value representing some quantity at that location. In our terrain example, this value is height. In weather maps, it might be temperature or pressure.

.. code-block:: python

   # A scalar field is just a 2D NumPy array
   height_map = np.zeros((100, 100))  # 100x100 grid of heights

   # Each position (row, col) has a height value
   height_map[50, 50] = 1.0  # Peak at center
   height_map[25, 75] = 0.5  # Smaller hill

NumPy's ``meshgrid`` function helps create coordinate grids that make it easy to compute values across the entire field at once [Harris2020]_:

.. code-block:: python

   # Create coordinate arrays
   x = np.linspace(-10, 10, 400)  # 400 points from -10 to 10
   y = np.linspace(-10, 10, 400)
   X, Y = np.meshgrid(x, y)  # 2D grids of x and y coordinates

   # Now X[i,j] gives the x-coordinate at position (i,j)
   # and Y[i,j] gives the y-coordinate

.. figure:: contour_concept.png
   :width: 500px
   :align: center
   :alt: Side-by-side comparison showing continuous gradient on left and stepped contour levels on right

   Left: Continuous scalar field. Right: Quantized into 8 discrete contour levels.

Stepped contours through quantization
--------------------------------------

**Quantization** converts continuous values into discrete levels. This is exactly what happens when you reduce the number of colors in an image, or when a topographic map uses fixed elevation bands.

The mathematical formula is straightforward:

.. code-block:: python

   # Quantize continuous values (0.0 to 1.0) into n discrete levels
   n_levels = 8
   level = (continuous_value * n_levels).astype(np.uint8)

   # Scale back to visible grayscale (0-255)
   pixel_value = level * (255 // n_levels)

When we apply this to an entire array, each "band" of heights maps to a single gray level, creating the characteristic stepped appearance of contour maps.

.. important::

   The number of levels controls the detail: fewer levels create broader bands (coarser contours), while more levels create finer gradations. Too many levels, and the steps become invisible.

Isolines using modulo operations
---------------------------------

An alternative to stepped bands is to draw thin **isolines**, the actual contour lines you see on traditional topographic maps. These mark specific height values.

The trick is to use the modulo operator (``%``) to find where height values cross regular intervals:

.. code-block:: python

   # Find pixels at specific height intervals
   # Scale height to 0-100, round, then check for multiples of 16
   isolines = ((height * 100).round() % 16) == 0

   # Result: True where height is at intervals, False elsewhere
   # Convert to image: True=255 (white line), False=0 (black)
   isolines = (isolines * 255).astype(np.uint8)

This creates thin white lines wherever the terrain crosses height thresholds, like the thin brown lines on a hiking map.

.. admonition:: Did You Know?

   Contour lines were first used systematically in the 1700s by French cartographers mapping the seabed. The technique was later applied to land elevation, revolutionizing how we represent 3D terrain on 2D maps [USGS2024]_. Today, the same mathematical principles are used in medical imaging (CT scans), weather forecasting (isobars), and even video game terrain rendering. Modern algorithms like Marching Cubes [Lorensen1987]_ extend contour extraction to 3D volumetric data.

Hands-On Exercises
==================

Now apply what you've learned with three progressively challenging exercises. This scaffolded approach [Sweller1988]_ builds understanding incrementally.

Exercise 1: Execute and explore
---------------------------------

**Time estimate:** 3 minutes

Run the main contour script to see both visualization techniques:

.. code-block:: python
   :caption: Run contour.py to generate both outputs
   :linenos:

   import numpy as np
   from PIL import Image

   # Create coordinate grid with extra dimension for multiple hills
   dim = np.linspace(-10, 10, 400)
   x, y, _ = np.meshgrid(dim, dim, [1])

   # Define three hills with different positions and sizes
   position_x = np.array([-3.0, 7.0, 9.0])
   position_y = np.array([0.0, 8.0, -9.0])
   width_x = np.array([5.3, 8.3, 4.0])
   width_y = np.array([6.3, 5.7, 4.0])

   # Calculate height as sum of Gaussians
   d = np.sqrt(((x - position_x) / width_x) ** 2 + ((y - position_y) / width_y) ** 2)
   z = np.exp(-d ** 2).sum(axis=2)
   znorm = (z - z.min()) / (z.max() - z.min())

   # Stepped contours (8 levels)
   contour = (znorm * 8).astype(np.uint8) * 32
   Image.fromarray(contour, mode='L').save('contour_steps.png')

   # Isolines
   isolines = ((znorm * 100).round() % 16) == 0
   Image.fromarray((isolines * 255).astype(np.uint8), mode='L').save('contour_isolines.png')

.. figure:: contour_steps.png
   :width: 350px
   :align: center
   :alt: Stepped contour showing three overlapping hills with 8 gray levels

   Stepped contour bands showing three Gaussian hills

.. figure:: contour_isolines.png
   :width: 350px
   :align: center
   :alt: Isoline visualization showing thin white contour lines on black background

   Isolines marking height intervals (thin white lines)

**Reflection questions:**

* How many distinct gray levels can you count in the stepped image?
* Why do the isolines appear as thin rings around the hill peaks?
* What happens where two hills overlap?

.. dropdown:: Answers
   :class: note

   **Gray levels:** 8 distinct levels (including black for the lowest areas)

   **Thin rings:** Each ring marks a specific height threshold. The modulo operation only returns True at exact intervals, creating thin lines rather than bands.

   **Overlapping hills:** Where hills overlap, their heights add together, creating higher combined peaks. You can see this where contours from different hills merge.

Exercise 2: Modify parameters
-------------------------------

**Time estimate:** 3-4 minutes

Modify the code from Exercise 1 to achieve these goals:

**Goal A:** Change the number of contour levels from 8 to 16

.. dropdown:: Hint
   :class: tip

   Look for where ``8`` appears in the stepped contour calculation. You'll also need to adjust the scaling factor (currently ``32``) to keep values in the 0-255 range.

.. dropdown:: Solution A
   :class: note

   .. code-block:: python

      # Change from 8 levels to 16 levels
      n_levels = 16
      contour = (znorm * n_levels).astype(np.uint8) * (255 // n_levels)

   With 16 levels, ``255 // 16 = 15``, so each level gets 15 intensity values. The bands become finer and more detailed.

**Goal B:** Move one hill to a different position

.. dropdown:: Hint
   :class: tip

   Modify the ``position_x`` and ``position_y`` arrays. Try changing the first hill from ``(-3.0, 0.0)`` to ``(0.0, 0.0)`` to center it.

.. dropdown:: Solution B
   :class: note

   .. code-block:: python

      # Center the first hill at the origin
      position_x = np.array([0.0, 7.0, 9.0])  # Changed -3.0 to 0.0
      position_y = np.array([0.0, 8.0, -9.0])  # First hill already at y=0

**Goal C:** Create wider, more spread-out hills

.. dropdown:: Hint
   :class: tip

   The ``width_x`` and ``width_y`` arrays control how spread out each hill is. Larger values create broader, flatter hills.

.. dropdown:: Solution C
   :class: note

   .. code-block:: python

      # Make all hills much wider
      width_x = np.array([8.0, 10.0, 7.0])  # Increased from original values
      width_y = np.array([9.0, 8.0, 7.0])

   Wider hills create smoother, more gradual slopes and fewer distinct contour bands.

Exercise 3: Create random terrain
-----------------------------------

**Time estimate:** 5 minutes

Now create your own terrain with randomly positioned hills. Use the starter code below and fill in the TODO sections.

**Goal:** Generate 3-5 hills at random positions with random sizes

**Requirements:**

* Use ``np.random.uniform()`` to generate random values
* Position hills within the grid bounds (-8 to 8)
* Width values between 2.0 and 6.0
* Output a stepped contour visualization

.. code-block:: python
   :caption: Starter code - fill in the blanks
   :linenos:

   import numpy as np
   from PIL import Image

   np.random.seed(42)  # For reproducibility

   # Create coordinate grid
   size = 400
   dim = np.linspace(-10, 10, size)
   x, y, _ = np.meshgrid(dim, dim, [1])

   # TODO: Set number of hills (between 3 and 5)
   num_hills = ___

   # TODO: Generate random positions
   position_x = np.random.uniform(___, ___, num_hills)
   position_y = np.random.uniform(___, ___, num_hills)

   # TODO: Generate random widths
   width_x = np.random.uniform(___, ___, num_hills)
   width_y = np.random.uniform(___, ___, num_hills)

   # Calculate terrain (provided)
   d = np.sqrt(((x - position_x) / width_x) ** 2 + ((y - position_y) / width_y) ** 2)
   z = np.exp(-d ** 2).sum(axis=2)
   znorm = (z - z.min()) / (z.max() - z.min())

   # TODO: Choose number of contour levels
   n_levels = ___
   contour = (znorm * n_levels).astype(np.uint8) * (255 // n_levels)

   Image.fromarray(contour, mode='L').save('random_terrain.png')

.. dropdown:: Hint 1: Random positions
   :class: tip

   Use ``np.random.uniform(-8, 8, num_hills)`` to generate positions within bounds. The first argument is the minimum, second is maximum, third is how many values to generate.

.. dropdown:: Hint 2: Random widths
   :class: tip

   Use ``np.random.uniform(2.0, 6.0, num_hills)`` for widths. Values below 2.0 create very sharp peaks; values above 6.0 create very flat, spread-out hills.

.. dropdown:: Complete Solution
   :class: note

   .. code-block:: python
      :linenos:
      :emphasize-lines: 11,14,15,18,19,27

      import numpy as np
      from PIL import Image

      np.random.seed(42)

      # Create coordinate grid
      size = 400
      dim = np.linspace(-10, 10, size)
      x, y, _ = np.meshgrid(dim, dim, [1])

      num_hills = 4  # Between 3 and 5

      # Random positions within grid bounds
      position_x = np.random.uniform(-8, 8, num_hills)
      position_y = np.random.uniform(-8, 8, num_hills)

      # Random widths for variety
      width_x = np.random.uniform(2.0, 6.0, num_hills)
      width_y = np.random.uniform(2.0, 6.0, num_hills)

      # Calculate terrain
      d = np.sqrt(((x - position_x) / width_x) ** 2 + ((y - position_y) / width_y) ** 2)
      z = np.exp(-d ** 2).sum(axis=2)
      znorm = (z - z.min()) / (z.max() - z.min())

      # Create contour with 10 levels
      n_levels = 10
      contour = (znorm * n_levels).astype(np.uint8) * (255 // n_levels)

      Image.fromarray(contour, mode='L').save('random_terrain.png')
      print(f"Created terrain with {num_hills} random hills")

.. figure:: random_terrain.png
   :width: 350px
   :align: center
   :alt: Random terrain with 4 hills showing contour bands

   Example output: Random terrain with 4 hills

**Challenge extension:** Create a "valley" effect by using negative Gaussian values, or try subtracting one hill from another to create a depression in the terrain.

Summary
=======

In this module, you learned how to create contour visualizations from scalar field data:

**Key takeaways:**

1. **Scalar fields** are 2D arrays where each value represents a quantity (height, temperature, etc.)
2. **Stepped contours** use quantization: ``(value * n_levels).astype(int)`` creates discrete bands
3. **Isolines** use modulo: ``(value * scale) % interval == 0`` finds specific thresholds
4. **Gaussian functions** create smooth "hills": ``exp(-(distance**2))`` falls off from center [Weisstein2024]_

**Common pitfalls:**

.. warning::

   * **Forgetting to normalize:** Always scale values to 0-1 before quantization, or you may get unexpected results
   * **Wrong dtype:** Use ``np.uint8`` for image output (0-255 range)
   * **Level count mismatch:** If you change the number of levels, adjust the scaling factor too (``255 // n_levels``)

References
==========

.. [Gonzalez2018] Gonzalez, R.C. and Woods, R.E. (2018). *Digital Image Processing* (4th ed.). Pearson. ISBN: 978-0-13-335672-4. [Chapter 10 on image segmentation and contour detection]

.. [Harris2020] Harris, C.R., et al. (2020). "Array programming with NumPy." *Nature*, 585, 357-362. https://doi.org/10.1038/s41586-020-2649-2

.. [Lorensen1987] Lorensen, W.E. and Cline, H.E. (1987). "Marching cubes: A high resolution 3D surface construction algorithm." *ACM SIGGRAPH Computer Graphics*, 21(4), 163-169. [Historical algorithm for 3D contour extraction]

.. [USGS2024] U.S. Geological Survey. (2024). "Topographic Maps." *USGS*. Retrieved December 7, 2024, from https://www.usgs.gov/programs/national-geospatial-program/topographic-maps [History and methodology of contour mapping]

.. [PillowDocs] Clark, A. (2015). *Pillow (PIL Fork) Documentation*. https://pillow.readthedocs.io/

.. [Weisstein2024] Weisstein, E.W. (2024). "Gaussian Function." *MathWorld*, Wolfram Research. https://mathworld.wolfram.com/GaussianFunction.html

.. [Sweller1988] Sweller, J. (1988). "Cognitive load during problem solving: Effects on learning." *Cognitive Science*, 12(2), 257-285. [Theoretical foundation for scaffolded learning approach]
