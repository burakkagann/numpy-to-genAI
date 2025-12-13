=====================================
3.4.4 - Fourier Art
=====================================

:Duration: 18 minutes
:Level: Intermediate
:Prerequisites: Module 3.4.1-3.4.3 (Signal Processing basics), NumPy arrays

Overview
========

What if you could see the hidden frequencies inside an image and manipulate them to create stunning visual effects? The Fast Fourier Transform (FFT) reveals a secret world within every picture: the frequency domain. Here, smooth gradients become low-frequency signals clustered at the center, while sharp edges become high-frequency components scattered at the edges.

In this exercise, you will learn to transform images into their frequency representation, apply creative filters, and reconstruct altered versions. By mastering frequency domain manipulation, you can create everything from subtle blurs to dramatic glitch art effects that would be impossible with pixel-by-pixel operations alone.

**Learning Objectives**

- Understand how the Fast Fourier Transform converts images from spatial to frequency domain
- Visualize and interpret frequency magnitude spectra
- Apply low-pass, high-pass, and band-pass filters to create artistic effects
- Create glitch art by manipulating specific frequency components

Quick Start: Visualize the Frequency Domain
===========================================

Let us start by seeing what an image looks like in the frequency domain. Run this code to create a checkerboard pattern and visualize its FFT:

.. code-block:: python
   :caption: simple_fft.py
   :linenos:

   import numpy as np
   from PIL import Image

   # Step 1: Create a checkerboard pattern
   size = 256
   image = np.zeros((size, size), dtype=np.float64)
   checker_size = 16
   for y in range(size):
       for x in range(size):
           if ((x // checker_size) + (y // checker_size)) % 2 == 0:
               image[y, x] = 255

   # Step 2: Apply 2D FFT and shift zero frequency to center
   fft_result = np.fft.fft2(image)
   fft_shifted = np.fft.fftshift(fft_result)

   # Step 3: Calculate magnitude spectrum (log scale for visibility)
   magnitude = np.abs(fft_shifted)
   magnitude_log = np.log1p(magnitude)
   magnitude_normalized = (magnitude_log / magnitude_log.max() * 255).astype(np.uint8)

   # Step 4: Create side-by-side comparison
   output = np.zeros((size, size * 2), dtype=np.uint8)
   output[:, :size] = image.astype(np.uint8)
   output[:, size:] = magnitude_normalized

   result = Image.fromarray(output, mode='L')
   result.save('simple_fft_output.png')

.. figure:: simple_fft_output.png
   :width: 600px
   :align: center
   :alt: Left side shows a checkerboard pattern, right side shows its FFT magnitude spectrum with bright dots

   Left: Original checkerboard pattern. Right: Frequency magnitude spectrum showing bright dots that correspond to the checkerboard's regular spacing.

.. tip::

   The bright center of the spectrum represents low frequencies (smooth areas), while the outer regions represent high frequencies (edges and details). The distinct dots in the spectrum correspond to the checkerboard's periodic structure.

Core Concepts
=============

Concept 1: The Frequency Domain
-------------------------------

The Fast Fourier Transform (FFT) is a mathematical operation that decomposes an image into its constituent frequencies, much like a prism splits white light into its component colors [CooleyTukey1965]_. While the original image exists in the spatial domain (where each pixel represents brightness at a location), the FFT converts it to the frequency domain (where each point represents the strength of a particular frequency pattern).

Understanding the Magnitude Spectrum
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When we apply FFT to an image, we get complex numbers containing both magnitude and phase information. For visualization, we typically look at the magnitude spectrum:

- **Center (low frequencies)**: Represents smooth, slowly-varying content like gradients and overall brightness
- **Edges (high frequencies)**: Represents rapidly-changing content like sharp edges and fine details
- **Bright spots**: Indicate strong presence of specific frequency patterns

NumPy provides efficient FFT functions [NumPyFFT]_:

.. code-block:: python

   # The key FFT operations in NumPy:
   fft_result = np.fft.fft2(image)      # 2D FFT
   fft_shifted = np.fft.fftshift(fft_result)  # Center the zero frequency
   magnitude = np.abs(fft_shifted)      # Get magnitude (ignore phase)

.. admonition:: Did You Know?

   The FFT algorithm, published by Cooley and Tukey in 1965, reduced the computational complexity from O(n squared) to O(n log n), making frequency analysis practical for real-world applications [CooleyTukey1965]_. This breakthrough is considered one of the most important algorithms of the 20th century.

Concept 2: Frequency Filtering
------------------------------

Once an image is in the frequency domain, we can selectively keep or remove certain frequencies by multiplying with a mask. This is the foundation of many image processing effects [GonzalezWoods2018]_.

Types of Frequency Filters
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Low-Pass Filter (Blur)**

Keeps low frequencies (center of spectrum), removes high frequencies (edges). The result is a blurred image because sharp transitions are removed.

.. code-block:: python

   # Create circular low-pass mask
   mask = np.zeros((size, size))
   center = size // 2
   radius = 30  # Smaller radius = more blur

   for y in range(size):
       for x in range(size):
           if np.sqrt((x-center)**2 + (y-center)**2) <= radius:
               mask[y, x] = 1.0

**High-Pass Filter (Edge Enhancement)**

Keeps high frequencies (edges of spectrum), removes low frequencies (center). The result shows only edges and fine details.

.. code-block:: python

   # High-pass is the inverse of low-pass
   mask = np.zeros((size, size))
   for y in range(size):
       for x in range(size):
           if np.sqrt((x-center)**2 + (y-center)**2) > radius:
               mask[y, x] = 1.0

**Band-Pass Filter**

Keeps only a specific range of frequencies between an inner and outer radius. This can isolate particular patterns or textures.

.. figure:: frequency_effects_comparison.png
   :width: 500px
   :align: center
   :alt: 2x2 grid showing original image, blur effect, edge enhancement, and band-pass result

   Comparison of frequency filtering effects. Top-left: Original. Top-right: Low-pass blur. Bottom-left: High-pass edges. Bottom-right: Band-pass.

.. important::

   To apply a filter, multiply the FFT by the mask, then use the inverse FFT to return to the spatial domain:

   .. code-block:: python

      filtered_fft = fft_shifted * mask
      result = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_fft)))

Concept 3: Glitch Art Through Frequency Manipulation
----------------------------------------------------

Glitch art embraces digital errors as aesthetic elements [Menkman2011]_. By deliberately corrupting the frequency domain in creative ways, we can produce visually striking effects that would be difficult to achieve through traditional image manipulation.

Glitch Techniques in the Frequency Domain
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Asymmetric Filtering**

Apply different filters to different parts of the spectrum. For example, blur the left half while sharpening the right half, creating surreal split-reality effects.

**Random Frequency Zeroing**

Randomly zero out rectangular regions in the frequency domain. This creates "missing data" artifacts similar to corrupted digital files.

**Phase Scrambling**

While magnitude determines the strength of frequencies, phase determines their alignment. Scrambling phase information while preserving magnitude creates dreamlike, ghostly effects.

.. code-block:: python

   # Example: Adding noise to phase in high-frequency regions
   magnitude = np.abs(fft_shifted)
   phase = np.angle(fft_shifted)

   # Add random noise to phase
   phase_noise = np.random.uniform(-0.5, 0.5, phase.shape)
   modified_phase = phase + phase_noise

   # Reconstruct with original magnitude, modified phase
   glitched = magnitude * np.exp(1j * modified_phase)

.. note::

   Glitch art has roots in the 1960s experimental music scene and gained popularity in visual art during the 2000s. Artists like Rosa Menkman have established glitch as a legitimate artistic practice that questions our relationship with technology [MoradiScott2009]_.

Hands-On Exercises
==================

Exercise 1: Execute and Explore (3-4 minutes)
---------------------------------------------

Run ``simple_fft.py`` to visualize the FFT of a checkerboard pattern.

.. code-block:: bash

   python simple_fft.py

**Reflection Questions:**

1. Why do you think there are distinct bright dots in the magnitude spectrum rather than a smooth distribution?
2. What do you predict would happen to the spectrum if you changed ``checker_size`` from 16 to 8?
3. The center of the spectrum is brightest. What does this tell you about the image content?

.. dropdown:: Solution and Explanation

   **Answers:**

   1. The bright dots appear because the checkerboard is a periodic pattern with a specific frequency. Each dot corresponds to a harmonic of the checkerboard's fundamental frequency. Regular patterns produce discrete frequency peaks.

   2. With ``checker_size = 8``, the checkerboard would have higher frequency (smaller squares = faster variation). The bright dots would move further from the center toward the edges of the spectrum.

   3. The bright center indicates significant low-frequency content, representing the overall average brightness level of the image. Even a high-contrast checkerboard has a DC component (average gray level).

Exercise 2: Modify Filter Parameters (4-5 minutes)
--------------------------------------------------

Use the ``frequency_filter.py`` script to experiment with different filter settings.

**Goals:**

1. Change the filter radius from 30 to 15 to see stronger blur
2. Change the filter radius to 50 to see subtle blur
3. Modify the code to create a band-pass filter (keep frequencies between radius 20 and 60)

**Starter code location:** ``frequency_filter.py``

.. dropdown:: Hints

   **Hint 1:** For a band-pass filter, you need two conditions: distance must be greater than the inner radius AND less than or equal to the outer radius.

   **Hint 2:** The mask creation loop looks like:

   .. code-block:: python

      for y in range(size):
          for x in range(size):
              dist = np.sqrt((x - center)**2 + (y - center)**2)
              if inner_radius < dist <= outer_radius:
                  mask[y, x] = 1.0

.. dropdown:: Complete Band-Pass Solution

   .. code-block:: python
      :linenos:

      def create_bandpass_mask(size, inner_radius=20, outer_radius=60):
          """Create a band-pass frequency mask."""
          mask = np.zeros((size, size), dtype=np.float64)
          center = size // 2

          for y in range(size):
              for x in range(size):
                  dist = np.sqrt((x - center)**2 + (y - center)**2)
                  if inner_radius < dist <= outer_radius:
                      mask[y, x] = 1.0

          return mask

      # Apply the band-pass filter
      bandpass_mask = create_bandpass_mask(size, inner_radius=20, outer_radius=60)
      bandpass_result = apply_frequency_filter(original, bandpass_mask)

   The band-pass filter removes both the very low frequencies (overall brightness) and very high frequencies (fine noise), keeping only the middle-range patterns.

Exercise 3: Create Glitch Art from Scratch (5-6 minutes)
--------------------------------------------------------

Create your own glitch art effect by completing ``glitch_art_starter.py``.

**Requirements:**

1. Create an asymmetric frequency mask (different effect on left vs right halves)
2. Add at least 5 "glitch holes" (zeroed frequency regions)
3. Save the result showing original and glitched images side-by-side

**Starter code:** ``glitch_art_starter.py``

.. code-block:: python
   :caption: Key section to complete
   :linenos:

   # TODO 1: Create asymmetric mask
   mask = np.ones((size, size), dtype=np.float64)
   center = size // 2

   for y in range(size):
       for x in range(size):
           dist = np.sqrt((x - center)**2 + (y - center)**2)
           if x < center:
               # Left side: YOUR RULE HERE (e.g., low-pass)
               mask[y, x] = ???
           else:
               # Right side: YOUR RULE HERE (e.g., high-pass)
               mask[y, x] = ???

   # TODO 2: Add glitch holes
   np.random.seed(42)
   for i in range(5):
       gx = np.random.randint(0, size)
       gy = np.random.randint(0, size)
       hole_size = ???
       mask[???:???, ???:???] = 0

.. dropdown:: Hint 1: Asymmetric Mask Logic

   For the left side, try keeping low frequencies (blur effect):

   .. code-block:: python

      if x < center:
          mask[y, x] = 1.0 if dist < 40 else 0.1

   For the right side, try keeping high frequencies (edge effect):

   .. code-block:: python

      else:
          mask[y, x] = 0.1 if dist < 20 else 1.0

.. dropdown:: Hint 2: Glitch Holes

   To create glitch holes, pick random positions and set rectangular regions to zero:

   .. code-block:: python

      hole_size = np.random.randint(5, 15)
      y_start = max(0, gy - hole_size)
      y_end = min(size, gy + hole_size)
      x_start = max(0, gx - hole_size)
      x_end = min(size, gx + hole_size)
      mask[y_start:y_end, x_start:x_end] = 0

.. dropdown:: Complete Solution

   See ``glitch_art_solution.py`` for the full implementation. The key insights:

   .. code-block:: python
      :linenos:

      # Asymmetric mask: blur left, sharpen right
      for y in range(size):
          for x in range(size):
              dist = np.sqrt((x - center)**2 + (y - center)**2)
              if x < center:
                  mask[y, x] = 1.0 if dist < 40 else 0.1
              else:
                  mask[y, x] = 0.1 if dist < 20 else 1.0

      # Add 15 random glitch holes
      np.random.seed(42)
      for _ in range(15):
          gx = np.random.randint(0, size)
          gy = np.random.randint(0, size)
          hole_size = np.random.randint(5, 15)
          y_start, y_end = max(0, gy-hole_size), min(size, gy+hole_size)
          x_start, x_end = max(0, gx-hole_size), min(size, gx+hole_size)
          mask[y_start:y_end, x_start:x_end] = 0

.. figure:: glitch_art_output.png
   :width: 600px
   :align: center
   :alt: Side-by-side comparison of original geometric pattern and glitched version with asymmetric distortion

   Example output showing original pattern (left) and glitched result (right) with asymmetric filtering and frequency holes.

**Challenge Extension:**

Create an animated glitch effect by gradually changing the mask parameters over multiple frames. Save as a GIF to see your glitch art in motion.

Summary
=======

In this exercise, you learned to see images from a completely new perspective: the frequency domain. The FFT reveals the hidden harmonic structure of images, allowing you to manipulate content in ways impossible through pixel operations alone.

**Key Takeaways:**

- The FFT converts images from spatial domain to frequency domain [OppenheimSchafer2010]_
- Low frequencies (center of spectrum) represent smooth content; high frequencies (edges) represent details
- Frequency filters (low-pass, high-pass, band-pass) selectively modify image content [BrighamFFT]_
- Glitch art uses creative frequency manipulation to produce intentional "errors" as aesthetic elements

For deeper understanding of digital signal processing and FFT applications, see [Smith1997]_.

**Common Pitfalls to Avoid:**

- Forgetting to use ``fftshift`` before visualization (zero frequency will be at corners, not center)
- Not using log scale for magnitude display (frequency magnitudes vary enormously)
- Modifying phase without understanding its effect (can produce unexpected ghosting)

References
==========

.. [CooleyTukey1965] Cooley, J. W., & Tukey, J. W. (1965). An algorithm for the machine calculation of complex Fourier series. *Mathematics of Computation*, 19(90), 297-301. https://doi.org/10.1090/S0025-5718-1965-0178586-1

.. [GonzalezWoods2018] Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing* (4th ed.). Pearson. ISBN: 978-0-13-335672-4 [Chapter 4: Filtering in the Frequency Domain]

.. [NumPyFFT] NumPy Developers. (2024). Discrete Fourier Transform (numpy.fft). *NumPy Documentation*. Retrieved December 7, 2025, from https://numpy.org/doc/stable/reference/routines.fft.html

.. [Menkman2011] Menkman, R. (2011). *The Glitch Moment(um)*. Institute of Network Cultures. ISBN: 978-90-816021-6-7 [Foundational text on glitch art theory]

.. [MoradiScott2009] Moradi, I., & Scott, A. (2009). *Glitch: Designing Imperfection*. Mark Batty Publisher. ISBN: 978-0-9795546-3-2

.. [OppenheimSchafer2010] Oppenheim, A. V., & Schafer, R. W. (2010). *Discrete-Time Signal Processing* (3rd ed.). Pearson. ISBN: 978-0-13-198842-2 [Chapters on DFT and FFT]

.. [Smith1997] Smith, S. W. (1997). *The Scientist and Engineer's Guide to Digital Signal Processing*. California Technical Publishing. Available free at https://www.dspguide.com/ [Excellent FFT tutorial]

.. [BrighamFFT] Brigham, E. O. (1988). *The Fast Fourier Transform and Its Applications*. Prentice Hall. ISBN: 978-0-13-307505-2 [Comprehensive FFT reference]
