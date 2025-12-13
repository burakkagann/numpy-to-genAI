.. _module-3-3-5-delaunay:

=====================================
3.3.5 - Delaunay Triangulation
=====================================

:Duration: 18 minutes
:Level: Intermediate
:Prerequisites: Module 2.1 (Basic Shapes), NumPy array basics, matplotlib fundamentals

Overview
========

Delaunay triangulation is a fundamental technique in computational geometry that connects a set of points into triangles with remarkable properties. Named after the Russian mathematician Boris Delaunay who formalized it in 1934, this algorithm ensures that no point lies inside the circumcircle of any triangle, which naturally produces well-shaped triangles that avoid thin, elongated "slivers."

In generative art, Delaunay triangulation enables the creation of striking low-poly aesthetics, organic mosaic patterns, and mesh-based visualizations. By sampling colors from source images and filling triangles with averaged values, you can transform photographs into abstract geometric art with a distinctive crystalline appearance.

**Learning Objectives**

By completing this module, you will:

* Understand what Delaunay triangulation is and why it produces high-quality triangular meshes
* Use ``scipy.spatial.Delaunay`` to triangulate arbitrary point sets in Python
* Create colorful geometric art by filling triangles with sampled colors
* Transform procedural images into low-poly art style using color averaging techniques

Quick Start: Your First Triangulation
=====================================

Let's start by seeing Delaunay triangulation in action. Run this code to generate a triangular mesh from random points:

.. code-block:: python
   :caption: Simple Delaunay triangulation
   :linenos:

   import numpy as np
   import matplotlib.pyplot as plt
   from scipy.spatial import Delaunay

   # Generate random points
   np.random.seed(42)
   points = np.random.rand(50, 2) * 400

   # Compute Delaunay triangulation
   triangulation = Delaunay(points)

   # Visualize
   plt.figure(figsize=(8, 8))
   plt.triplot(points[:, 0], points[:, 1], triangulation.simplices,
               color='steelblue', linewidth=0.8, linestyle='-')
   plt.plot(points[:, 0], points[:, 1], 'o', color='coral', markersize=6)
   plt.axis('equal')
   plt.axis('off')
   plt.savefig('simple_delaunay.png', dpi=150, bbox_inches='tight')
   plt.close()

.. figure:: simple_delaunay.png
   :width: 500px
   :align: center
   :alt: A wireframe triangulation of 50 random points showing connected triangles

   Delaunay triangulation of 50 random points creates 87 non-overlapping triangles that fill the convex hull.

.. tip::

   Notice how the triangles have relatively uniform shapes. Delaunay triangulation maximizes the minimum angle of all triangles, which avoids the long, thin "sliver" triangles that would result from naive triangulation approaches.

What is Delaunay Triangulation?
===============================

The Circumcircle Property
-------------------------

The defining characteristic of Delaunay triangulation is the **circumcircle property**: for every triangle in the mesh, no other point from the point set lies inside that triangle's circumcircle (the unique circle passing through all three vertices).

**Circumcircle** is the circle that passes through all three vertices of a triangle. The center of this circle is equidistant from all three vertices.

.. code-block:: python

   def circumcircle(p1, p2, p3):
       """Calculate circumcircle center and radius for a triangle."""
       ax, ay = p1
       bx, by = p2
       cx, cy = p3

       d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
       ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) +
             (cx**2 + cy**2) * (ay - by)) / d
       uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) +
             (cx**2 + cy**2) * (bx - ax)) / d
       radius = np.sqrt((ax - ux)**2 + (ay - uy)**2)
       return (ux, uy), radius

.. important::

   The circumcircle property is not just mathematical elegance; it has practical consequences. By ensuring no point lies inside any circumcircle, Delaunay triangulation maximizes the minimum angle across all triangles, producing meshes that are numerically stable for computations and visually pleasing for art.

.. admonition:: Did You Know?

   Boris Delaunay first described this triangulation in his 1934 paper "Sur la sphere vide" (On the empty sphere). The same principle extends to 3D, where tetrahedra have circumspheres instead of circumcircles.

   (Delaunay, 1934)

.. figure:: delaunay_concept_diagram.png
   :width: 600px
   :align: center
   :alt: Four-panel diagram showing Delaunay triangulation concepts

   (a) Basic triangulation, (b) circumcircle property demonstration, (c) grid-based point distribution, (d) triangle quality colored by minimum angle.

Point Distribution Strategies
=============================

How Points Affect Results
-------------------------

The placement of sample points dramatically affects the resulting triangulation and artistic output. Different strategies create different visual aesthetics:

**Random distribution** creates organic, natural-looking patterns with varied triangle sizes. This works well for abstract art and texture generation.

**Grid-based distribution** (with optional jitter) produces more regular, structured patterns. Adding small random offsets prevents perfectly aligned triangles while maintaining overall uniformity.

**Edge-weighted distribution** places more points along image edges and high-contrast regions. This preserves important features when creating low-poly art from photographs.

.. code-block:: python

   # Random distribution
   points_random = np.random.rand(100, 2) * [width, height]

   # Grid with jitter
   gx, gy = np.meshgrid(np.linspace(0, width, 10),
                        np.linspace(0, height, 10))
   points_grid = np.column_stack([gx.ravel(), gy.ravel()])
   points_grid += np.random.randn(*points_grid.shape) * 5  # Add jitter

.. note::

   For low-poly art effects, start with 100-200 random points for a moderate level of abstraction. Fewer points (50-80) create more dramatic geometric shapes, while more points (300+) preserve more detail from the source image.

Creating Low-Poly Art
=====================

From Colors to Triangles
------------------------

The magic of low-poly art comes from sampling colors from a source image and using those colors to fill the Delaunay triangles. The simplest approach samples the color at each triangle's centroid.

.. code-block:: python

   from matplotlib.collections import PolyCollection

   def get_triangle_color(source, vertices):
       """Sample color at triangle centroid."""
       centroid = np.mean(vertices, axis=0)
       cx = int(np.clip(centroid[0], 0, source.shape[1] - 1))
       cy = int(np.clip(centroid[1], 0, source.shape[0] - 1))
       # Note: image indexing is [y, x], not [x, y]!
       return source[cy, cx] / 255.0

   # Build colored triangles
   triangles = [points[s] for s in triangulation.simplices]
   colors = [get_triangle_color(image, points[s])
             for s in triangulation.simplices]

   # Render with PolyCollection
   collection = PolyCollection(triangles, facecolors=colors, edgecolors='none')

.. figure:: colored_triangulation.png
   :width: 500px
   :align: center
   :alt: Colorful triangulation with randomly colored filled triangles

   Random colors filling Delaunay triangles create abstract geometric art.

.. important::

   Remember that NumPy image arrays use ``[row, column]`` indexing, which corresponds to ``[y, x]`` in spatial coordinates. This is a common source of bugs when sampling colors from images!

Hands-On Exercises
==================

These exercises follow the Execute, Modify, Re-code progression to build your triangulation skills step by step.

Exercise 1: Execute and Explore
-------------------------------

**Time estimate:** 3-4 minutes

Run the ``simple_delaunay.py`` script and observe the triangulation output. Pay attention to the triangle shapes and how they connect the points.

.. code-block:: python
   :caption: Exercise 1 - Run and observe
   :linenos:

   import numpy as np
   import matplotlib.pyplot as plt
   from scipy.spatial import Delaunay

   np.random.seed(42)
   num_points = 50
   points = np.random.rand(num_points, 2) * 400

   triangulation = Delaunay(points)

   plt.figure(figsize=(8, 8))
   plt.triplot(points[:, 0], points[:, 1], triangulation.simplices,
               color='steelblue', linewidth=0.8, linestyle='-')
   plt.plot(points[:, 0], points[:, 1], 'o', color='coral', markersize=6)
   plt.title(f'{num_points} points, {len(triangulation.simplices)} triangles')
   plt.axis('equal')
   plt.axis('off')
   plt.savefig('simple_delaunay.png', dpi=150, bbox_inches='tight')
   plt.close()

   print(f"Points: {num_points}, Triangles: {len(triangulation.simplices)}")

**Reflection questions:**

* How many triangles were created from 50 points? Can you find a pattern?
* Why do the triangles only cover the "convex hull" of the points rather than a rectangular region?
* Do you notice any extremely thin or elongated triangles? Why or why not?

.. dropdown:: Solution & Explanation

   **Answer:** 50 points create approximately 87-90 triangles (the exact number depends on point positions).

   **Why:**

   * For N points with H points on the convex hull, Delaunay triangulation creates approximately 2N - 2 - H triangles.
   * The triangulation only covers the convex hull because Delaunay connects points without introducing new vertices.
   * There are no thin slivers because Delaunay maximizes minimum angles by the circumcircle property.

Exercise 2: Modify to Achieve Goals
-----------------------------------

**Time estimate:** 3-4 minutes

Modify the colored triangulation to create different visual effects.

**Goals:**

1. Create a triangulation with exactly 200 colored triangles (adjust num_points)
2. Add white edges between triangles for a stained-glass effect
3. Use a gradient color scheme instead of random colors

.. dropdown:: Hints

   * For goal 1: More points means more triangles. Try increasing ``num_points`` until you get close to 200 triangles.
   * For goal 2: Change ``edgecolors='none'`` to ``edgecolors='white'`` and set ``linewidths=1``.
   * For goal 3: Instead of ``np.random.rand(3)``, calculate color based on position, e.g., ``[x/width, y/height, 0.5]``.

.. dropdown:: Solutions

   **1. Create ~200 triangles:**

   .. code-block:: python

      num_points = 100  # Approximately doubles the triangle count
      # With corners added, this creates roughly 200 triangles

   **2. Stained-glass effect:**

   .. code-block:: python

      collection = PolyCollection(triangles, facecolors=colors,
                                  edgecolors='white', linewidths=1.5)

   **3. Gradient colors:**

   .. code-block:: python

      for simplex in triangulation.simplices:
          triangle = points[simplex]
          centroid = np.mean(triangle, axis=0)
          # Color based on position
          color = [centroid[0]/width, centroid[1]/height, 0.5]
          colors.append(color)

Exercise 3: Create Low-Poly Art from Scratch
--------------------------------------------

**Time estimate:** 5-6 minutes

Create a low-poly art effect by triangulating a procedural image and filling triangles with sampled colors.

**Goal:** Transform a generated gradient/pattern image into a geometric low-poly style artwork.

**Requirements:**

* Generate a colorful procedural source image (gradient, circles, or patterns)
* Create 150+ sample points across the image
* Triangulate the points and fill each triangle with its centroid color
* Display both original and low-poly versions side by side

**Hints:**

* Include corner points ``[[0,0], [W,0], [W,H], [0,H]]`` to cover the entire image
* Remember: image arrays use ``[y, x]`` indexing, not ``[x, y]``
* Use ``plt.ylim(HEIGHT, 0)`` to flip the Y-axis for correct image orientation

.. code-block:: python
   :caption: Exercise 3 starter code

   import numpy as np
   import matplotlib.pyplot as plt
   from matplotlib.collections import PolyCollection
   from scipy.spatial import Delaunay

   WIDTH, HEIGHT = 400, 400

   # TODO: Create a procedural source image (gradient or pattern)
   source_image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

   # TODO: Generate sample points (random + corners)
   np.random.seed(42)
   points = None  # Your code here

   # TODO: Compute triangulation
   triangulation = None  # Your code here

   # TODO: Sample colors for each triangle
   triangles = []
   colors = []
   # Your code here

   # TODO: Visualize with PolyCollection
   # Your code here

.. dropdown:: Complete Solution

   .. code-block:: python
      :caption: Low-poly art complete solution
      :linenos:
      :emphasize-lines: 20-24, 40-48

      import numpy as np
      import matplotlib.pyplot as plt
      from matplotlib.collections import PolyCollection
      from scipy.spatial import Delaunay

      WIDTH, HEIGHT = 400, 400

      # Create procedural source image
      source_image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
      y, x = np.ogrid[:HEIGHT, :WIDTH]
      source_image[:, :, 0] = np.clip(255 - y * 0.4, 0, 255).astype(np.uint8)
      source_image[:, :, 1] = np.clip(100 + np.sin(x * 0.02) * 80, 0, 255).astype(np.uint8)
      source_image[:, :, 2] = np.clip(50 + y * 0.5, 0, 255).astype(np.uint8)

      # Add sun circle
      cx, cy = WIDTH // 2, HEIGHT // 3
      dist = np.sqrt((x - cx)**2 + (y - cy)**2)
      sun_mask = dist < 80
      source_image[sun_mask] = [255, 220, 50]

      # Generate sample points
      np.random.seed(42)
      points = np.random.rand(150, 2) * [WIDTH, HEIGHT]
      corners = np.array([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]])
      points = np.vstack([points, corners])

      # Compute triangulation
      triangulation = Delaunay(points)

      # Sample colors for each triangle
      triangles = []
      colors = []
      for simplex in triangulation.simplices:
          triangle = points[simplex]
          triangles.append(triangle)
          centroid = np.mean(triangle, axis=0)
          cx = int(np.clip(centroid[0], 0, WIDTH - 1))
          cy = int(np.clip(centroid[1], 0, HEIGHT - 1))
          colors.append(source_image[cy, cx] / 255.0)

      # Visualize
      fig, axes = plt.subplots(1, 2, figsize=(12, 6))
      axes[0].imshow(source_image)
      axes[0].set_title('Original')
      axes[0].axis('off')

      collection = PolyCollection(triangles, facecolors=colors, edgecolors='none')
      axes[1].add_collection(collection)
      axes[1].set_xlim(0, WIDTH)
      axes[1].set_ylim(HEIGHT, 0)
      axes[1].set_aspect('equal')
      axes[1].set_title('Low-Poly Art')
      axes[1].axis('off')

      plt.tight_layout()
      plt.savefig('lowpoly_art_output.png', dpi=150, bbox_inches='tight')

   **How it works:**

   * Lines 20-24 generate random points plus corner points to ensure full coverage
   * Lines 40-48 iterate through triangles, sampling color at each centroid
   * The centroid is the average of the three vertices: ``np.mean(triangle, axis=0)``
   * Y-axis is flipped with ``set_ylim(HEIGHT, 0)`` to match image coordinates

   **Challenge extension:** Try adding edge detection to place more points along high-contrast boundaries for better feature preservation!

.. figure:: lowpoly_art_output.png
   :width: 600px
   :align: center
   :alt: Side-by-side comparison of original procedural image and low-poly triangulated version

   Example output showing the procedural source image (left) transformed into low-poly art (right).

Summary
=======

In just 18 minutes, you have learned how to create triangular meshes from point sets and transform images into geometric art:

**Key takeaways:**

* Delaunay triangulation maximizes minimum angles, avoiding thin sliver triangles
* The circumcircle property ensures no point lies inside any triangle's circumcircle
* ``scipy.spatial.Delaunay`` makes triangulation simple: just pass an array of points
* Low-poly art is created by sampling colors at triangle centroids and filling with PolyCollection

**Common pitfalls to avoid:**

* Forgetting corner points leaves triangulation only covering the convex hull, not the full canvas
* Using ``[x, y]`` indexing for images instead of ``[y, x]`` (row, column) ordering
* Not flipping the Y-axis when rendering results in an upside-down image

This knowledge of mesh generation prepares you for more advanced topics like Voronoi diagrams, terrain generation, and 3D mesh processing.

References
==========

.. [Delaunay1934] Delaunay, B. (1934). Sur la sphere vide. *Bulletin de l'Academie des Sciences de l'URSS*, 6, 793-800. [Original paper establishing Delaunay triangulation]

.. [deBerg2008] de Berg, M., Cheong, O., van Kreveld, M., & Overmars, M. (2008). *Computational Geometry: Algorithms and Applications* (3rd ed.). Springer. [Comprehensive textbook on computational geometry]

.. [Fortune1987] Fortune, S. (1987). A sweepline algorithm for Voronoi diagrams. *Algorithmica*, 2(1), 153-174. [Efficient algorithm for computing Voronoi/Delaunay]

.. [Shewchuk2002] Shewchuk, J. R. (2002). Delaunay refinement algorithms for triangular mesh generation. *Computational Geometry*, 22(1-3), 21-74. [Mesh quality and refinement]

.. [Aurenhammer1991] Aurenhammer, F. (1991). Voronoi diagrams: A survey of a fundamental geometric data structure. *ACM Computing Surveys*, 23(3), 345-405. [Voronoi-Delaunay duality explained]

.. [SciPyDocs] SciPy Developers. (2024). scipy.spatial.Delaunay. *SciPy Documentation*. https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html [Official API reference]

.. [Shiffman2012] Shiffman, D. (2012). *The Nature of Code*. Self-published. https://natureofcode.com/ [Creative coding applications of computational geometry]
