.. _lines:

=====================================
2.1.1 - Drawing Lines
=====================================

:Duration: 18 minutes
:Level: Beginner
:Prerequisites: Module 1.1.1 (RGB Color Basics), Module 0.3.1 (Images as Data)

.. contents:: Contents
   :local:
   :depth: 2

Overview
========

Lines are the fundamental building blocks of computer graphics. From the earliest vector displays to modern GPUs, the ability to draw a line between two points remains essential. But how do we represent something continuous—like a mathematical line—using discrete pixels?

In this exercise, you will learn how computers draw lines by interpolating points between endpoints. You will implement line drawing using NumPy, understand the mathematics behind it, and discover how simple lines can create complex generative art patterns. By the end, you will see how geometric primitives become artistic tools.

**Learning Objectives**

By completing this exercise, you will:

* Understand line drawing algorithms and their historical context (Bresenham, DDA)
* Implement parametric line interpolation using NumPy's ``linspace``
* Create generative line art patterns through iteration
* Recognize lines as fundamental primitives in computational geometry

Quick Start: Draw Your First Line
==================================

Let's start by drawing a single diagonal line across a canvas. This demonstrates the core concept: a line is simply a series of connected pixels.

.. code-block:: python
   :caption: simple_line.py - Draw a diagonal line
   :linenos:
   :emphasize-lines: 13,17-18,22

   import numpy as np
   from PIL import Image

   # Create blank canvas (grayscale image)
   canvas = np.zeros((400, 400), dtype=np.uint8)

   # Define line endpoints
   x_start, y_start = 50, 50
   x_end, y_end = 350, 350

   # Calculate number of points needed
   # At least one point per pixel in the longer dimension
   num_points = max(abs(x_end - x_start), abs(y_end - y_start)) + 1

   # Generate interpolated coordinates using linspace
   # linspace creates evenly spaced points between start and end
   x_coords = np.linspace(x_start, x_end, num_points).round().astype(int)
   y_coords = np.linspace(y_start, y_end, num_points).round().astype(int)

   # Draw line by setting pixels to white (255)
   # Remember: array indexing is [row, column] which is [y, x]
   canvas[y_coords, x_coords] = 255

   # Save result
   Image.fromarray(canvas).save('simple_line.png')

.. figure:: simple_line.png
   :width: 400px
   :align: center
   :alt: A white diagonal line on black background

   A simple diagonal line from (50, 50) to (350, 350). Notice how the continuous mathematical line is represented as discrete white pixels on a black canvas.

.. tip::

   The key insight: Lines are **interpolated points**. We calculate enough points to fill every pixel along the line's path, ensuring no gaps appear.

Core Concept 1: Line Drawing Algorithms
========================================

The Challenge of Discrete Lines
--------------------------------

In mathematics, a line is defined by the equation :math:`y = mx + b` or parametrically as :math:`(x(t), y(t))`. These are **continuous** functions—they exist at every point along the infinite real number line. But computer screens are **discrete grids** of pixels with integer coordinates.

How do we convert a continuous line into discrete pixels? This is the fundamental challenge of **rasterization**: converting vector graphics (mathematical descriptions) into raster graphics (pixel grids).

**Historical Approaches**

The earliest computers faced this challenge when creating vector displays and pen plotters. Two famous algorithms emerged:

1. **DDA (Digital Differential Analyzer)**: Incremental algorithm that steps through one coordinate and calculates the other. Simple but requires floating-point arithmetic.

2. **Bresenham's Line Algorithm**: Integer-only algorithm invented in 1962 by Jack Bresenham for IBM plotters. Uses only addition, subtraction, and bit shifts—crucial for early hardware with no floating-point units [Bresenham1965]_.

.. admonition:: Did You Know?

   Bresenham's algorithm was invented for pen plotters—mechanical devices that physically drew on paper. The algorithm needed to be extremely efficient because it controlled physical motors [Bresenham1965]_.

NumPy's Linspace as Interpolator
---------------------------------

In modern Python, we do not need to implement Bresenham's algorithm from scratch (though it is instructive to do so). Instead, NumPy provides ``linspace``—a function that creates evenly-spaced points along a line.

.. code-block:: python

   # Create 5 evenly-spaced points from 0 to 10
   points = np.linspace(0, 10, 5)
   # Result: array([0.0, 2.5, 5.0, 7.5, 10.0])

When applied to both x and y coordinates, ``linspace`` effectively implements the **parametric line equation**:

.. math::

   x(t) = x_0 + t(x_1 - x_0)

   y(t) = y_0 + t(y_1 - y_0)

where :math:`t \in [0, 1]` is the parameter that varies from start (0) to end (1).

**Why Round and Convert to Int?**

``linspace`` returns floating-point values, but pixel indices must be integers. We use ``round()`` before ``astype(int)`` to ensure proper rounding:

.. code-block:: python

   x_coords = np.linspace(50, 350, 301).round().astype(int)

Without ``round()``, ``astype(int)`` would truncate (e.g., 2.9 → 2), causing visual artifacts.

.. important::

   **Always round before converting to int** when generating pixel coordinates. Truncation creates gaps; proper rounding ensures smooth lines.

Core Concept 2: Parametric Representation
==========================================

Mathematics of Interpolation
-----------------------------

The parametric line equation is beautiful in its simplicity. Given two points :math:`P_0 = (x_0, y_0)` and :math:`P_1 = (x_1, y_1)`, every point on the line can be expressed as:

.. math::

   P(t) = P_0 + t(P_1 - P_0) = (1-t)P_0 + tP_1

where :math:`t \in [0, 1]`.

* When :math:`t = 0`: We are at :math:`P_0` (start)
* When :math:`t = 0.5`: We are at the midpoint
* When :math:`t = 1`: We are at :math:`P_1` (end)

This is called **linear interpolation** or "lerp" for short. It is fundamental to computer graphics, animation, and numerical methods [Foley1990]_.

Code Implementation
-------------------

NumPy's ``linspace`` directly implements this:

.. code-block:: python

   # Create 301 evenly-spaced t values from 0 to 1
   t_values = np.linspace(0, 1, 301)

   # Calculate x and y using parametric equations
   x_coords = x_start + t_values * (x_end - x_start)
   y_coords = y_start + t_values * (y_end - y_start)

In practice, ``linspace`` simplifies this:

.. code-block:: python

   # Equivalent to the above, but cleaner
   x_coords = np.linspace(x_start, x_end, 301)
   y_coords = np.linspace(y_start, y_end, 301)

**How Many Points?**

Too few points create gaps; too many waste memory. The optimal number is:

.. code-block:: python

   num_points = max(abs(x_end - x_start), abs(y_end - y_start)) + 1

This ensures at least one point per pixel along the longer dimension [GonzalezWoods2018]_.

Core Concept 3: Lines as Generative Art
========================================

Historical Context
------------------

While line drawing was born from engineering necessity, artists quickly recognized its creative potential. Early computer art pioneers used lines as their primary expressive tool:

* **Vera Molnár** (1960s-present): Used plotters to create geometric line compositions, exploring systematically varied parameters [Molnar1974]_.

* **Sol LeWitt** (1960s-2000s): Created "wall drawings" based on simple line-drawing instructions executed by others—a conceptual approach that parallels algorithmic art [LeWitt1967]_.

* **Naum Gabo** (1920s-1970s): Though working with physical materials, created sculptural "constructions" using strings and lines that anticipate computational line art.

The connection is profound: **algorithms are instructions**, just as LeWitt's wall drawings were instructions. The computer becomes the artist's assistant, executing simple rules to create complex results.

From Utility to Aesthetics
---------------------------

Let's see how iteration transforms a single line into a pattern:

.. figure:: line_pattern.png
   :width: 400px
   :align: center
   :alt: Radial pattern of lines emanating from a single point

   A radial line pattern created by drawing lines from a fixed point (50, 200) to evenly-spaced points on the opposite edge. Simple iteration creates visual complexity.

The code behind this pattern:

.. code-block:: python

   fixed_x, fixed_y = 50, 200
   target_x = 350

   for target_y in range(0, 400, 50):
       draw_line(canvas, fixed_x, fixed_y, target_x, target_y)

**Key Insight**: Iteration + variation = emergent pattern. We are not drawing the pattern directly; we are defining rules that generate it.

Hands-On Exercises
==================

Exercise 1: Execute and Explore
--------------------------------

**Time estimate:** 3 minutes

Run ``simple_line.py`` and observe the output. This introduces you to the basic line drawing function.

**Reflection Questions:**

* Why do we need ``round()`` before converting to int?
* What happens if you swap the start and end points?
* Why use ``max()`` when calculating ``num_points``?

.. dropdown:: Solution & Explanation

   **Answer to Question 1:** ``round()`` is needed because ``linspace`` returns floating-point numbers (e.g., 50.0, 50.5, 51.0), but array indices must be integers. Without rounding, ``astype(int)`` would truncate (always round down), causing 50.9 → 50 instead of → 51. This creates visual gaps in diagonal lines.

   **Answer to Question 2:** Swapping start and end points produces the **same visual line** because ``linspace`` generates points in order from start to end. The line from (50, 50) to (350, 350) looks identical to the line from (350, 350) to (50, 50).

   **Answer to Question 3:** We use ``max()`` because:

   * Horizontal line (y constant): Need points equal to x distance
   * Vertical line (x constant): Need points equal to y distance
   * Diagonal line: Need points equal to the **larger** of x or y distance

   Taking the max ensures we have enough points regardless of line orientation.

Exercise 2: Modify Parameters
------------------------------

**Time estimate:** 4 minutes

Modify ``simple_line.py`` to achieve these goals:

**Goals:**

1. Draw a horizontal line from (50, 200) to (350, 200)
2. Draw a vertical line from (200, 50) to (200, 350)
3. Draw 5 parallel diagonal lines spaced 40 pixels apart

.. dropdown:: Hints

   **Hint for Goal 1:** Only change the endpoint coordinates. For a horizontal line, both y-values should be the same.

   **Hint for Goal 2:** For a vertical line, both x-values should be the same.

   **Hint for Goal 3:** Use a ``for`` loop. Draw the first line from (50, 50) to (350, 350), then increase both start and end y-coordinates by 40 for each subsequent line.

.. dropdown:: Solutions

   **1. Horizontal Line:**

   .. code-block:: python

      x_start, y_start = 50, 200
      x_end, y_end = 350, 200  # Same y-coordinate

   This creates a straight horizontal line across the center of the canvas.

   **2. Vertical Line:**

   .. code-block:: python

      x_start, y_start = 200, 50
      x_end, y_end = 200, 350  # Same x-coordinate

   This creates a straight vertical line down the center.

   **3. Five Parallel Diagonal Lines:**

   .. code-block:: python

      canvas = np.zeros((400, 400), dtype=np.uint8)

      for i in range(5):
          offset = i * 40
          x_start, y_start = 50, 50 + offset
          x_end, y_end = 350, 350 + offset

          num_points = max(abs(x_end - x_start), abs(y_end - y_start)) + 1
          x_coords = np.linspace(x_start, x_end, num_points).round().astype(int)
          y_coords = np.linspace(y_start, y_end, num_points).round().astype(int)

          canvas[y_coords, x_coords] = 255

   This creates 5 parallel lines, each offset by 40 pixels vertically.

Exercise 3: Create a Sunburst Pattern
--------------------------------------

**Time estimate:** 6 minutes

Create a "sunburst" pattern: lines radiating from the center to evenly-spaced points on the canvas edge.

**Goal:** Create a radial pattern with at least 16 lines emanating from the center (200, 200) to points on a circle.

**Requirements:**

* Canvas size: 400×400 pixels
* Center point: (200, 200)
* At least 16 evenly-spaced rays
* Use trigonometry to calculate endpoints

**Hints:**

* Use ``np.linspace`` to create evenly-spaced angles from 0 to :math:`2\pi`
* Convert polar coordinates to Cartesian: :math:`x = r \cos(\theta), y = r \sin(\theta)`
* Loop through angles, drawing a line from center to each calculated endpoint
* Remember to round and convert to integers

.. code-block:: python
   :caption: Exercise 3 starter code

   import numpy as np
   from PIL import Image

   def draw_line(canvas, x_start, y_start, x_end, y_end):
       # (Copy the draw_line function from previous examples)
       pass

   canvas = np.zeros((400, 400), dtype=np.uint8)
   center_x, center_y = 200, 200
   radius = 180  # Distance from center to edge

   # Your code here: create sunburst pattern
   # Step 1: Create array of angles
   # Step 2: Loop through angles
   # Step 3: Calculate endpoint using cos/sin
   # Step 4: Draw line from center to endpoint

   Image.fromarray(canvas).save('sunburst.png')

.. dropdown:: Complete Solution

   .. code-block:: python
      :caption: Sunburst Pattern Solution
      :linenos:
      :emphasize-lines: 8,13-16

      import numpy as np
      from PIL import Image

      def draw_line(canvas, x_start, y_start, x_end, y_end):
          num_points = max(abs(x_end - x_start) + 1, abs(y_end - y_start) + 1)
          x_coords = np.linspace(x_start, x_end, num_points).round().astype(int)
          y_coords = np.linspace(y_start, y_end, num_points).round().astype(int)
          canvas[y_coords, x_coords] = 255

      canvas = np.zeros((400, 400), dtype=np.uint8)
      center_x, center_y = 200, 200
      radius = 180
      num_rays = 24  # More rays = denser pattern

      # Create evenly-spaced angles from 0 to 2π
      angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)

      # Draw rays using polar-to-Cartesian conversion
      for angle in angles:
          # Convert polar (angle, radius) to Cartesian (x, y)
          end_x = int(center_x + radius * np.cos(angle))
          end_y = int(center_y + radius * np.sin(angle))

          # Draw line from center to calculated endpoint
          draw_line(canvas, center_x, center_y, end_x, end_y)

      Image.fromarray(canvas).save('sunburst.png')

   **How it works:**

   * **Line 8**: ``np.linspace(0, 2 * np.pi, num_rays, endpoint=False)`` creates evenly-spaced angles. ``endpoint=False`` ensures the last angle is not exactly :math:`2\pi` (which would duplicate the first angle at 0).

   * **Lines 13-14**: Polar-to-Cartesian conversion. ``cos(angle)`` gives the x-component, ``sin(angle)`` gives the y-component. We multiply by ``radius`` to scale the unit circle to our desired size, then add ``center_x`` and ``center_y`` to translate from origin to center.

   * **Lines 15-16**: Draw line from center to the calculated endpoint.

   **Challenge extension:** Try creating a star pattern by alternating between two different radii! For example, every other ray could have ``radius = 90`` while others have ``radius = 180``.

.. figure:: sunburst_example.png
   :width: 400px
   :align: center
   :alt: Radial sunburst pattern with 24 evenly-spaced rays

   Example output: A sunburst pattern with 24 rays radiating from the center. This demonstrates how trigonometry and iteration create symmetrical generative art.

Summary
=======

In just 18 minutes, you have learned the fundamentals of line drawing in computer graphics.

**Key Takeaways:**

* Lines are **interpolated points** between start and end coordinates
* NumPy's ``linspace`` provides parametric interpolation: :math:`P(t) = (1-t)P_0 + tP_1`
* Always ``round()`` before converting to ``int`` to avoid visual artifacts
* Calculating ``max(abs(x_end - x_start), abs(y_end - y_start)) + 1`` points ensures smooth lines
* **Iteration + simple rules = complex patterns** (foundational principle of generative art)

**Common Pitfalls to Avoid:**

* **Forgetting to round:** ``astype(int)`` truncates, not rounds. Use ``.round().astype(int)`` to avoid gaps.
* **Confusing (x, y) vs (row, col):** Array indexing is ``array[row, col]`` which is ``array[y, x]``. Be careful with coordinate order!
* **Not enough points:** Using too few interpolated points creates gaps. Always calculate based on the line's length.
* **Using the wrong data type:** Images require ``dtype=np.uint8`` for 0-255 pixel values.

This foundational knowledge prepares you for more complex geometric primitives. Lines combine to form triangles, polygons, and curves. The parametric thinking you have learned here extends to Bézier curves, splines, and transformations.

Next Steps
==========

Continue to **Module 2.1.2: Triangles** to learn how three lines create closed shapes. You will explore triangle drawing, filling algorithms, and how triangles form the basis of 3D graphics.

:doc:`Continue to Module 2.1.2 - Triangles <../2.1.2_triangles/triangles/README>`

References
==========

.. [Bresenham1965] Bresenham, J. E. (1965). Algorithm for computer control of a digital plotter. *IBM Systems Journal*, 4(1), 25-30. https://doi.org/10.1147/sj.41.0025 [Original paper describing the famous integer-only line algorithm for pen plotters]

.. [NumPyLinspace] NumPy Developers. (2024). numpy.linspace. *NumPy Documentation*. Retrieved January 30, 2025, from https://numpy.org/doc/stable/reference/generated/numpy.linspace.html [Official documentation for NumPy's linear interpolation function]

.. [Molnar1974] Molnár, V. (1974). Toward aesthetic guidelines for paintings with the aid of a computer. *Leonardo*, 7(3), 185-189. https://doi.org/10.2307/1572906 [Pioneer of computer-generated geometric art discussing systematic exploration of visual parameters]

.. [LeWitt1967] LeWitt, S. (1967). Paragraphs on conceptual art. *Artforum*, 5(10), 79-83. [Foundational text on conceptual art where instructions/algorithms are the artwork]

.. [Foley1990] Foley, J. D., van Dam, A., Feiner, S. K., & Hughes, J. F. (1990). *Computer Graphics: Principles and Practice* (2nd ed.). Addison-Wesley. ISBN: 978-0201121100 [Chapter 3: Output Primitives - comprehensive coverage of line drawing algorithms]

.. [GonzalezWoods2018] Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing* (4th ed.). Pearson. ISBN: 978-0133356724 [Standard textbook covering image representation, interpolation, and geometric transformations]
