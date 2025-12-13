.. _module-3-1-3-nonlinear-distortions:

=====================================
3.1.3 - Nonlinear Distortions
=====================================

:Duration: 18-20 minutes
:Level: Intermediate
:Prerequisites: Modules 3.1.1-3.1.2

Overview
========

In this module, you will explore nonlinear distortions, a powerful technique for creating visually striking image effects. Unlike the linear transformations covered in previous modules (rotation, scaling, shearing), nonlinear distortions warp images in ways that vary across the image plane. You will create wave patterns, barrel distortions, and swirl effects using pure NumPy.

**Learning Objectives**

By completing this module, you will:

* Understand the concept of coordinate remapping for image distortion
* Implement wave distortion using sinusoidal displacement
* Create barrel (fish-eye) and swirl effects with radial transformations
* Combine multiple distortions for complex visual effects

Quick Start: Your First Wave Distortion
========================================

Let's start by creating a wave distortion effect. Run this code to see an image warp with a sinusoidal pattern:

.. code-block:: python
   :caption: Create a simple wave distortion
   :linenos:

   import numpy as np
   from PIL import Image

   # Create a colorful checkerboard pattern (400x400)
   size = 400
   tile_size = 50
   image = np.zeros((size, size, 3), dtype=np.uint8)

   colors = [(255, 100, 100), (100, 100, 255), (100, 255, 100), (255, 255, 100)]
   for row in range(size // tile_size):
       for col in range(size // tile_size):
           color = colors[(row + col) % len(colors)]
           y_start, y_end = row * tile_size, (row + 1) * tile_size
           x_start, x_end = col * tile_size, (col + 1) * tile_size
           image[y_start:y_end, x_start:x_end] = color

   # Apply horizontal wave distortion
   amplitude = 20  # How far pixels shift
   frequency = 3   # Number of wave cycles
   distorted = np.zeros_like(image)

   for y in range(size):
       for x in range(size):
           offset = int(amplitude * np.sin(2 * np.pi * frequency * y / size))
           source_x = (x + offset) % size
           distorted[y, x] = image[y, source_x]

   result = Image.fromarray(distorted)
   result.save('wave_distortion_output.png')

.. figure:: wave_distortion_output.png
   :width: 400px
   :align: center
   :alt: Checkerboard pattern with horizontal wave distortion applied

   A checkerboard pattern distorted by a horizontal sine wave. Notice how the vertical edges become wavy while horizontal edges remain straight.

.. tip::

   The key insight is **coordinate remapping**: for each output pixel, we calculate where to sample from the input image. The sine function creates the wavy displacement.

Core Concepts
=============

Coordinate Remapping Fundamentals
----------------------------------

Nonlinear distortions work by remapping pixel coordinates. For each pixel in the output image, we calculate which pixel from the input image should be copied there. This is called **inverse mapping** because we work backward from output to input.

.. code-block:: python

   # Basic remapping structure
   for y in range(height):
       for x in range(width):
           # Calculate source coordinates (the "remapping")
           source_x = some_function(x, y)
           source_y = another_function(x, y)

           # Copy pixel from source to output
           output[y, x] = input[source_y, source_x]

.. figure:: coordinate_remapping_diagram.png
   :width: 600px
   :align: center
   :alt: Diagram showing grid lines before and after wave distortion

   Coordinate remapping visualized: the left grid shows original coordinates, the right shows how vertical lines become curved after wave distortion.

.. important::

   We use **inverse mapping** (output to input) rather than forward mapping (input to output) because it guarantees every output pixel gets a value. Forward mapping can leave "holes" where no input pixel maps to.

Common Distortion Types
------------------------

There are three fundamental nonlinear distortion types, each using different coordinate remapping formulas:

**Wave Distortion**

Displaces pixels using a sinusoidal function. The displacement can be horizontal, vertical, or both:

.. code-block:: python

   # Horizontal wave: x shifts based on y
   offset = amplitude * np.sin(2 * np.pi * frequency * y / size)
   source_x = x + offset

**Barrel (Fish-Eye) Distortion**

Radially pushes pixels outward from the center. The effect increases with distance from center:

.. code-block:: python

   # Radial distortion
   radius = np.sqrt(dx**2 + dy**2)
   factor = 1 + strength * radius**2
   source_x = dx / factor + center_x
   source_y = dy / factor + center_y

**Swirl Distortion**

Rotates pixels around the center, with more rotation near the center:

.. code-block:: python

   # Swirl: angle decreases with radius
   twist = twist_amount * (1 - radius / max_radius)
   source_x = center_x + radius * np.cos(angle - twist)
   source_y = center_y + radius * np.sin(angle - twist)

.. figure:: wave_variations_grid.png
   :width: 600px
   :align: center
   :alt: 2x2 grid showing wave distortion with different amplitude and frequency settings

   Wave distortion with different parameters. Top row: low frequency. Bottom row: high frequency. Left column: low amplitude. Right column: high amplitude.

.. admonition:: Did You Know?

   Barrel distortion is named after the visual effect of looking through a barrel or fish-eye lens. It was first mathematically described in the context of camera lens correction by Brown in 1966 [Brown1966]_. Today, the same mathematics is used both to correct lens distortion in photographs and to intentionally create artistic effects.

Practical Implementation
-------------------------

When implementing distortions, you must handle two practical issues:

**Out-of-Bounds Coordinates**

When source coordinates fall outside the image, you have two options:

.. code-block:: python

   # Option 1: Wrap around (creates seamless tiling)
   source_x = source_x % size

   # Option 2: Clip to edge (prevents wrapping artifacts)
   source_x = np.clip(source_x, 0, size - 1)

**Nearest-Neighbor Sampling**

With integer pixel coordinates, we use nearest-neighbor sampling, which simply rounds to the nearest integer. This can create blocky edges at high distortion levels, but keeps the implementation simple:

.. code-block:: python

   source_x = int(calculated_x)  # Round to nearest integer
   source_y = int(calculated_y)

.. note::

   More advanced techniques like bilinear interpolation can produce smoother results, but require sampling from four neighboring pixels and computing weighted averages. For learning purposes, nearest-neighbor is sufficient and easier to understand.

Hands-On Exercises
==================

Now apply what you have learned with three progressively challenging exercises. Each builds on the previous one using the **Execute, Modify, Create** approach.

Exercise 1: Execute and Explore
--------------------------------

**Time estimate:** 3-4 minutes

Run the ``simple_wave_distortion.py`` script and observe the output. Study the code to understand how it works.

.. code-block:: python
   :caption: Exercise 1 - Run this script
   :linenos:

   import numpy as np
   from PIL import Image

   # Create checkerboard
   size = 400
   tile_size = 50
   image = np.zeros((size, size, 3), dtype=np.uint8)
   colors = [(255, 100, 100), (100, 100, 255), (100, 255, 100), (255, 255, 100)]
   for row in range(size // tile_size):
       for col in range(size // tile_size):
           color = colors[(row + col) % len(colors)]
           image[row*tile_size:(row+1)*tile_size, col*tile_size:(col+1)*tile_size] = color

   # Apply wave distortion
   amplitude = 20
   frequency = 3
   distorted = np.zeros_like(image)
   for y in range(size):
       for x in range(size):
           offset = int(amplitude * np.sin(2 * np.pi * frequency * y / size))
           source_x = (x + offset) % size
           distorted[y, x] = image[y, source_x]

   Image.fromarray(distorted).save('wave_output.png')

**Reflection questions:**

1. Why do the horizontal lines stay straight while vertical lines become wavy?
2. What happens at the left and right edges of the image?
3. How would increasing the frequency parameter change the result?

.. dropdown:: Solution & Explanation
   :class: note

   **1. Why horizontal lines stay straight:**

   The wave formula ``offset = amplitude * sin(y)`` only modifies the x-coordinate. For any given row (fixed y), all pixels shift by the same amount, so horizontal lines remain horizontal. Vertical lines become wavy because different rows have different offsets.

   **2. Edge behavior:**

   The modulo operator ``% size`` wraps coordinates that go past the edge back to the other side. This creates a seamless "tiling" effect at the edges.

   **3. Higher frequency:**

   Increasing frequency adds more wave cycles across the image height. With ``frequency = 6``, you would see twice as many wave periods compared to ``frequency = 3``.

Exercise 2: Modify to Achieve Goals
------------------------------------

**Time estimate:** 3-4 minutes

Modify the wave distortion code to achieve each of these goals:

**Goals:**

1. Create a **vertical wave** (horizontal lines become wavy, vertical lines stay straight)
2. Increase the wave **amplitude** to 40 pixels
3. Create a **combined effect** with both horizontal and vertical waves

.. dropdown:: Hint: Vertical Wave
   :class: tip

   To make a vertical wave, swap which coordinate gets the offset. Instead of modifying ``source_x`` based on ``y``, modify ``source_y`` based on ``x``.

.. dropdown:: Solutions
   :class: note

   **1. Vertical wave:**

   .. code-block:: python

      # Change this loop body:
      for y in range(size):
          for x in range(size):
              offset = int(amplitude * np.sin(2 * np.pi * frequency * x / size))
              source_y = (y + offset) % size  # Now y changes, not x
              distorted[y, x] = image[source_y, x]

   **2. Increased amplitude:**

   Simply change ``amplitude = 40`` to increase the wave height.

   **3. Combined waves:**

   .. code-block:: python

      h_amplitude, h_frequency = 15, 3
      v_amplitude, v_frequency = 15, 4

      for y in range(size):
          for x in range(size):
              h_offset = int(h_amplitude * np.sin(2 * np.pi * h_frequency * y / size))
              v_offset = int(v_amplitude * np.sin(2 * np.pi * v_frequency * x / size))
              source_x = (x + h_offset) % size
              source_y = (y + v_offset) % size
              distorted[y, x] = image[source_y, source_x]

Exercise 3: Create Combined Waves
----------------------------------

**Time estimate:** 5-6 minutes

Now create your own combined wave distortion from scratch. Your goal is to combine horizontal and vertical waves to create an interesting "wobbly" effect.

**Requirements:**

* Apply both horizontal and vertical wave distortions simultaneously
* Use different frequencies for each direction (creates more interesting patterns)
* Save the result as ``my_combined_waves.png``

**Starter code:**

.. code-block:: python
   :caption: Exercise 3 starter code
   :linenos:

   import numpy as np
   from PIL import Image

   # Create checkerboard
   size = 400
   tile_size = 50
   image = np.zeros((size, size, 3), dtype=np.uint8)
   colors = [(255, 100, 100), (100, 100, 255), (100, 255, 100), (255, 255, 100)]
   for row in range(size // tile_size):
       for col in range(size // tile_size):
           color = colors[(row + col) % len(colors)]
           image[row*tile_size:(row+1)*tile_size, col*tile_size:(col+1)*tile_size] = color

   # TODO: Define amplitude and frequency for horizontal wave
   # TODO: Define amplitude and frequency for vertical wave

   distorted = np.zeros_like(image)

   for y in range(size):
       for x in range(size):
           # TODO: Calculate horizontal offset based on y
           # TODO: Calculate vertical offset based on x
           # TODO: Apply both offsets to find source coordinates
           pass  # Replace with your implementation

   Image.fromarray(distorted).save('my_combined_waves.png')

.. dropdown:: Hint 1: Define Parameters
   :class: tip

   Start by defining parameters for both waves:

   .. code-block:: python

      h_amplitude = 15
      h_frequency = 3
      v_amplitude = 15
      v_frequency = 4

.. dropdown:: Hint 2: Calculate Offsets
   :class: tip

   Calculate both offsets inside the loop:

   .. code-block:: python

      h_offset = int(h_amplitude * np.sin(2 * np.pi * h_frequency * y / size))
      v_offset = int(v_amplitude * np.sin(2 * np.pi * v_frequency * x / size))

.. dropdown:: Complete Solution
   :class: note

   .. code-block:: python
      :linenos:
      :emphasize-lines: 15-18, 24-27

      import numpy as np
      from PIL import Image

      # Create checkerboard
      size = 400
      tile_size = 50
      image = np.zeros((size, size, 3), dtype=np.uint8)
      colors = [(255, 100, 100), (100, 100, 255), (100, 255, 100), (255, 255, 100)]
      for row in range(size // tile_size):
          for col in range(size // tile_size):
              color = colors[(row + col) % len(colors)]
              image[row*tile_size:(row+1)*tile_size, col*tile_size:(col+1)*tile_size] = color

      # Wave parameters
      h_amplitude = 15
      h_frequency = 3
      v_amplitude = 15
      v_frequency = 4

      distorted = np.zeros_like(image)

      for y in range(size):
          for x in range(size):
              h_offset = int(h_amplitude * np.sin(2 * np.pi * h_frequency * y / size))
              v_offset = int(v_amplitude * np.sin(2 * np.pi * v_frequency * x / size))
              source_x = (x + h_offset) % size
              source_y = (y + v_offset) % size
              distorted[y, x] = image[source_y, source_x]

      Image.fromarray(distorted).save('my_combined_waves.png')

   **How it works:** The horizontal wave shifts pixels left/right based on their y-position, while the vertical wave shifts pixels up/down based on their x-position. Using different frequencies (3 and 4) creates a more complex, non-repeating pattern.

.. figure:: combined_waves_output.png
   :width: 400px
   :align: center
   :alt: Checkerboard with combined horizontal and vertical wave distortion

   Example output with combined waves. The grid pattern appears to ripple in multiple directions simultaneously.

**Challenge extension:** Modify the formula to create a "ripple" effect emanating from the center, where amplitude decreases with distance from center.

Summary
=======

In this module, you have learned the fundamental techniques for creating nonlinear image distortions:

**Key takeaways:**

* Nonlinear distortions work by **remapping coordinates**, calculating source positions for each output pixel
* **Wave distortion** uses sinusoidal functions to create wavy displacement
* **Barrel distortion** radially pushes pixels outward from center using ``r^2`` scaling
* **Swirl distortion** rotates pixels by an angle that varies with distance from center
* Multiple distortions can be combined by applying multiple coordinate transformations

**Common pitfalls to avoid:**

* Forgetting to handle out-of-bounds coordinates (use modulo or clipping)
* Confusing forward and inverse mapping (always use inverse: output to input)
* Using floating-point coordinates without converting to integers for indexing
* Applying distortion to the output array instead of reading from input

This foundation in coordinate remapping prepares you for more advanced effects like kaleidoscopes, morphing, and lens simulations.

References
==========

.. [Brown1966] Brown, D. C. (1966). Decentering distortion of lenses. *Photogrammetric Engineering*, 32(3), 444-462. [Original mathematical formulation of radial lens distortion]

.. [GonzalezWoods2018_warp] Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing* (4th ed.). Pearson. ISBN: 978-0133356724. [Chapter 2 on geometric transformations and image warping]

.. [Wolberg1990] Wolberg, G. (1990). *Digital Image Warping*. IEEE Computer Society Press. ISBN: 0-8186-8944-7. [Comprehensive treatment of image warping algorithms]

.. [Szeliski2022] Szeliski, R. (2022). *Computer Vision: Algorithms and Applications* (2nd ed.). Springer. https://szeliski.org/Book/ [Section 3.6 on geometric transformations]

.. [Foley1990] Foley, J. D., van Dam, A., Feiner, S. K., & Hughes, J. F. (1990). *Computer Graphics: Principles and Practice* (2nd ed.). Addison-Wesley. [Chapter 5 on 2D transformations]

.. [Harris2020] Harris, C. R., et al. (2020). Array programming with NumPy. *Nature*, 585, 357-362. https://doi.org/10.1038/s41586-020-2649-2 [NumPy array indexing and operations]

.. [Clark2015] Clark, A. (2015). *Pillow (PIL Fork) Documentation*. https://pillow.readthedocs.io/ [Image I/O and manipulation with Python]
