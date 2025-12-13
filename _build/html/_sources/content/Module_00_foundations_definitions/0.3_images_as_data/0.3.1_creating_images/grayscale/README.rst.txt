.. _module-0-3-1-creating-images:

=====================================
0.3.1 - Creating Images
=====================================

:Duration: 10-12 minutes
:Level: Beginner
:Prerequisites: Module 0.2.1 - Defining AI, ML, and Algorithms

Overview
========

Digital images are arrays of numbers. Understanding this fundamental concept is essential for algorithmic art, machine learning, and AI powered art generation. In this module, you'll learn how to create grayscale images using NumPy arrays, setting the foundation for all image manipulation throughout this course.

.. figure:: /content/Module_00_foundations_definitions/0.3_images_as_data/0.3.1_creating_images/grayscale/grayscale.png
   :width: 400px
   :align: center
   :alt: A simple gray square created from a NumPy array

   A 200×200 grayscale image created with code

**Learning Objectives**

By completing this module, you will:

* Understand that images are 2D arrays of numerical values
* Create grayscale images using NumPy and Pillow (PIL)
* Manipulate array dimensions, shape, and data types correctly
* Recognize the relationship between array values (0-255) and brightness
* Write code that generates basic grayscale patterns

Quick Start: Your First Image
===============================

Let's create a simple grayscale image with just a few lines of code. Run this to see immediate results:

.. code-block:: python
   :caption: Create a gray square
   :linenos:

   import numpy as np
   from PIL import Image

   # Create a 200x200 array filled with the value 128
   array = np.zeros((200, 200), dtype=np.uint8)
   array += 128

   # Convert array to image and save
   image = Image.fromarray(array)
   image.save('gray.png')

**Result:** A medium-gray 200×200 pixel square saved as ``gray.png``.

.. tip::
   
   If you're using Jupyter Notebook, you can display the image directly by putting ``image`` in a cell by itself. No need to save the file first.

Understanding Images as Arrays
================================

The fundamental insight
-----------------------

**A digital image is a 2D grid of pixels, and each pixel has a brightness value.** In Python with NumPy, we represent this grid as a 2-dimensional array where each number corresponds to one pixel's intensity.

For grayscale images:
- **0** = black (no light)
- **255** = white (maximum light)
- **128** = medium gray (half intensity)

.. code-block:: python
   :caption: Image as array concept

   # This array represents a 3×3 grayscale image
   array = np.array([
       [0,   128, 255],  # Row 0: black, gray, white
       [64,  128, 192],  # Row 1: dark, gray, light
       [255, 128, 0  ]   # Row 2: white, gray, black
   ], dtype=np.uint8)

   # Convert to viewable image
   image = Image.fromarray(array)

.. figure:: /content/Module_00_foundations_definitions/0.3_images_as_data/0.3.1_creating_images/grayscale/lincoln.png
   :width: 700px
   :align: center
   :alt: Diagram showing how array values map to pixel brightness

   Array values directly map to pixel brightness: 0=black, 255=white, intermediate values=grays

Array dimensions and shape
---------------------------

NumPy arrays for grayscale images have **two dimensions**: height (rows) and width (columns).

.. important::
   
   **Shape notation:** ``(height, width)`` or equivalently ``(rows, columns)``
   
   Note that height comes first, not width! This follows matrix notation where vertical dimension precedes horizontal.

.. code-block:: python
   :caption: Creating arrays with specific dimensions

   # Create a 100 pixels tall, 200 pixels wide image
   array = np.zeros((100, 200), dtype=np.uint8)
   
   print(array.shape)  # Output: (100, 200)
   # 100 rows (height), 200 columns (width)

**Common shapes:**

* ``(200, 200)`` -> Square image, 200×200 pixels
* ``(480, 640)`` -> Rectangular image, 480 pixels tall, 640 pixels wide
* ``(1080, 1920)`` -> Full HD resolution (1920×1080, but height first!)

The ``uint8`` data type
-------------------------

The ``dtype=np.uint8`` parameter specifies the data type for array values.

**uint8** means:
- **u** = unsigned (no negative numbers)
- **int** = integer (whole numbers only)
- **8** = 8 bits per value

**Range:** 0 to 255 (2⁸ = 256 possible values)

.. code-block:: python
   :caption: Why uint8 matters

   # Correct: uint8 for standard images
   array = np.zeros((100, 100), dtype=np.uint8)
   array += 128  # Sets all pixels to gray
   
   # Wrong: without dtype specification
   array = np.zeros((100, 100))  # Defaults to float64
   array += 128
   # This creates floats (128.0), which PIL may not handle correctly

.. note::
   
   **Why 0-255?** This range comes from 8-bit color depth, the standard for digital images. Each pixel uses exactly 1 byte (8 bits) of memory. Modern displays and file formats (PNG, JPEG) expect this range.

Coordinate system and indexing
-------------------------------

Arrays use **[row, column]** indexing, where:
- **row** = y-coordinate (vertical position, 0 at top)
- **column** = x-coordinate (horizontal position, 0 at left)

.. code-block:: python
   :caption: Accessing specific pixels

   array = np.zeros((200, 300), dtype=np.uint8)
   
   # Set pixel at row 50, column 100 to white
   array[50, 100] = 255
   
   # Set top-left corner pixel to black
   array[0, 0] = 0
   
   # Set bottom-right corner pixel to gray
   array[199, 299] = 128

.. figure:: /content/Module_00_foundations_definitions/0.3_images_as_data/0.3.1_creating_images/grayscale/coordinate.png
   :width: 500px
   :align: center
   :alt: Coordinate system showing origin at top-left

   Image coordinate system: origin (0,0) is at the top-left corner, y increases downward

.. tip::
   
   **Remember:** ``array[y, x]`` not ``array[x, y]``
   
   This is opposite to many graphics systems (like canvas coordinates), but consistent with matrix notation used throughout NumPy.

Hands-On Exercises
===============

Exercise 1: Create a black square
----------------------------------

**Time estimate:** 2-3 minutes

Create a perfectly black 150×150 pixel square image.

**Your task:**

1. Create a NumPy array with the correct shape
2. Ensure all pixel values are 0 (black)
3. Convert to an image and display or save it

.. code-block:: python
   :caption: Starter code
   
   import numpy as np
   from PIL import Image
   
   # Your code here: create array and image
   
   # Display or save
   image.save('black_square.png')

.. dropdown:: 💡 Solution

   .. code-block:: python
      :caption: Complete solution
      
      import numpy as np
      from PIL import Image
      
      # Create 150×150 array filled with zeros (black)
      array = np.zeros((150, 150), dtype=np.uint8)
      
      # Convert to image
      image = Image.fromarray(array)
      
      # Save
      image.save('black_square.png')
   
   **Explanation:**
   
   * ``np.zeros()`` creates an array filled with 0 values
   * Shape ``(150, 150)`` creates a square: 150 rows × 150 columns
   * ``dtype=np.uint8`` ensures values are in 0-255 range
   * ``Image.fromarray()`` interprets the array as a grayscale image
   * All zeros = all black pixels

.. figure:: /content/Module_00_foundations_definitions/0.3_images_as_data/0.3.1_creating_images/grayscale/black_square.png
   :width: 300px
   :align: center
   :alt: Black Square

   150x150 Pixels Black Square

Exercise 2: Create a white image
---------------------------------

**Time estimate:** 2-3 minutes

Now create a completely white 200×200 pixel image.

**Your task:**

1. Create an array
2. Set all values to 255 (white)
3. Convert and save

**Hint:** You can create an array and then add a value to all elements, or use ``np.ones()`` and multiply by 255.

.. dropdown:: Solution (Two Approaches)

   **Approach 1: Using zeros and addition**
   
   .. code-block:: python
      
      import numpy as np
      from PIL import Image
      
      array = np.zeros((200, 200), dtype=np.uint8)
      array += 255  # Add 255 to every element
      
      image = Image.fromarray(array)
      image.save('white_square.png')
   
   **Approach 2: Using ones and multiplication**
   
   .. code-block:: python
      
      import numpy as np
      from PIL import Image
      
      array = np.ones((200, 200), dtype=np.uint8) * 255
      
      image = Image.fromarray(array)
      image.save('white_square.png')
   
   **Explanation:**
   
   * ``np.ones()`` creates an array filled with 1 values
   * Multiplying by 255 scales all values to maximum brightness
   * Both approaches produce identical results
   * Choose based on clarity or preference

.. figure:: /content/Module_00_foundations_definitions/0.3_images_as_data/0.3.1_creating_images/grayscale/white_square.png
   :width: 300px
   :align: center
   :alt: White Square

   200x200 Pixels Black Square


Exercise 3: Create a rectangular gradient
------------------------------------------

**Time estimate:** 4-5 minutes

Create a 100×300 pixel image (100 tall, 300 wide) with a horizontal gradient from black (left) to white (right).

**Your task:**

1. Create a rectangular array (not square!)
2. Use a loop to vary brightness from left to right
3. Calculate brightness proportionally to column position

**Hints:**

* Loop over columns: ``for col in range(width):``
* Calculate brightness: ``brightness = col * 255 // width``
* Set entire column to that brightness: ``array[:, col] = brightness``

.. dropdown:: Solution

   .. code-block:: python
      :caption: Horizontal gradient solution
      :linenos:
      :emphasize-lines: 9-10
      
      import numpy as np
      from PIL import Image
      
      # Create rectangular array
      height, width = 100, 300
      array = np.zeros((height, width), dtype=np.uint8)
      
      # Fill with gradient
      for col in range(width):
          brightness = col * 255 // width
          array[:, col] = brightness
      
      # Convert and save
      image = Image.fromarray(array)
      image.save('gradient.png')
   
   **How it works:**
   
   * **Line 5:** Creates 100×300 array (shorter and wider)
   * **Line 9:** Loop iterates over all 300 columns
   * **Line 10:** Calculates brightness proportionally
     
     - When ``col = 0`` (left edge): ``brightness = 0`` (black)
     - When ``col = 299`` (right edge): ``brightness ≈ 255`` (white)
     - Middle columns: intermediate grays
   
   * **Line 11:** Sets entire column to calculated brightness
     
     - ``array[:, col]`` means "all rows in column col"
     - This sets all 100 pixels in that column at once
   
   **Result:** Smooth left-to-right gradient

.. figure:: /content/Module_00_foundations_definitions/0.3_images_as_data/0.3.1_creating_images/grayscale/gradient_.png
   :width: 400px
   :align: center
   :alt: Gradient

   100x300 Pixels Horizontal Gradient from Black to White

Summary
=======

In this module, you've learned the fundamental concept that powers all digital image manipulation:

**Key takeaways:**

* **Images are arrays:** Digital images are 2D NumPy arrays of numerical values
* **Grayscale values:** 0 = black, 255 = white, intermediate values = shades of gray
* **Array shape:** Specified as ``(height, width)`` where height (rows) comes first
* **Data type:** Use ``dtype=np.uint8`` for standard 8-bit images (0-255 range)
* **Coordinate system:** ``array[row, column]`` where (0,0) is top-left corner
* **Pillow conversion:** ``Image.fromarray()`` converts NumPy arrays to displayable images

**Why this matters:**

This foundational understanding unlocks algorithmic image creation. Remember from Module 0.2.1 that **algorithms are explicit rules**—now you can write rules that create visual outputs. Every generative art technique in this course builds on this principle: manipulating arrays to create images.

Next Steps
==========

Now that you understand grayscale images as arrays, you're ready to:

* **Module 0.3.2** — Settiing up the environment
* **Module 1.1.1** — Extend to RGB color images with 3-channel arrays
* **Module 1.1.2** — Learn color theory spaces

Continue to **Module 0.4.1: Settiing up the environments** to configure your local environment for next modules.

References
==========

.. [NumPy2020] Harris, C.R., et al. "Array programming with NumPy." Nature 585 (2020): 357-362. https://doi.org/10.1038/s41586-020-2649-2

.. [Pillow2024] Clark, A. "Pillow (PIL Fork) Documentation." 2024. https://pillow.readthedocs.io/

.. [Gonzalez2007] Gonzalez, R.C. and Woods, R.E. "Digital Image Processing." 3rd ed. Pearson, 2007. [Chapter 2 on digital image fundamentals]

.. [vanRossum2023] van Rossum, G. "The Python Tutorial: NumPy Arrays." Python Software Foundation, 2023. https://docs.python.org/3/tutorial/

.. [Matplotlib2007] Hunter, J.D. "Matplotlib: A 2D graphics environment." Computing in Science & Engineering 9.3 (2007): 90-95.