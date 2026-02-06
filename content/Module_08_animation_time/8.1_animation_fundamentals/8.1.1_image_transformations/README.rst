8.1.1 - Image Transformations
====================

Comprehensive image processing and transformation examples using NumPy and SciPy.

Overview
--------

This demo showcases various image manipulation techniques using the Python logo as a base image. It demonstrates fundamental concepts in digital image processing and computer graphics.

Features
--------

- Color manipulation and channel operations
- Geometric transformations (rotation, flipping)
- Blur and displacement effects
- Masking and shape operations
- Random noise effects
- Mathematical shape generation

Running the Demo
----------------

Make sure you have the required dependencies:

.. code-block:: bash

   pip install scipy numpy pillow

You'll need a ``python_logo.png`` file in the same directory. Then run:

.. code-block:: bash

   python transform_logo.py

Output Images
-------------

The script generates multiple output images demonstrating different effects:

- ``dim.png`` - Dimmed version
- ``flip.png`` - Horizontally flipped
- ``purple.png`` - Color channel manipulation
- ``rotate.png`` - Rotated image
- ``blur.png`` - Displacement blur effect
- ``roll.png`` - Channel displacement
- ``rand.png`` - Random noise overlay
- ``spaced.png`` - Spaced pixel effects
- ``square.png`` - Square masking
- ``circle.png`` - Mathematical circle generation
- ``logocircle.png`` - Logo with circle effect
- ``masked.png`` - Donut-shaped masking

Technical Details
-----------------

The demo covers:

- **Array operations**: Using NumPy for efficient image manipulation
- **Color space**: RGB channel manipulation
- **Geometric transforms**: Rotation and flipping
- **Convolution effects**: Blur and displacement
- **Mathematical shapes**: Circle and donut generation using coordinate grids
- **Masking**: Boolean operations for selective image modification

Learning Objectives
-------------------

After running this demo, you'll understand:

- How images are represented as NumPy arrays
- Basic color space manipulation
- Geometric transformations
- Mathematical shape generation
- Masking and selective operations
- File I/O for image processing
