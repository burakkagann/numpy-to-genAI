.. _module-11-2-3-face-detection:

================================
11.2.3 Face Detection
================================

:Duration: 35-40 minutes
:Level: Intermediate
:Prerequisites: Module 3.3.5 (Delaunay Triangulation)

Overview
========

Face detection is one of the most compelling applications of computer vision, enabling creative interactions between humans and digital systems. In this exercise, you will use MediaPipe's Face Mesh to detect 478 facial landmarks and transform them into striking low-poly portrait art using Delaunay triangulation.

This exercise connects directly to :ref:`Module 3.3.5 Delaunay Triangulation <module-3-3-5-delaunay>`, applying those geometric concepts to a new creative context. By combining face detection with triangulation, you will create a pipeline that transforms any face photograph into stylized geometric art, demonstrating how foundational algorithms transfer to advanced applications.

Learning Objectives
-------------------

By the end of this exercise, you will be able to:

* Distinguish between face detection (locating faces) and face recognition (identifying who)
* Use MediaPipe Face Mesh to detect 478 facial landmarks in real-time
* Apply Delaunay triangulation to facial landmarks for low-poly art generation
* Create interactive face-driven generative effects using webcam input


Quick Start: See It In Action
=============================

Run this code to transform a face photo into low-poly art:

.. code-block:: python
   :caption: Generate low-poly face art in under 20 lines
   :linenos:

   import cv2
   import numpy as np
   from scipy.spatial import Delaunay
   import mediapipe as mp
   from mediapipe.tasks import python
   from mediapipe.tasks.python import vision

   # Load image and detect face landmarks
   image = cv2.cvtColor(cv2.imread("sample_face.jpg"), cv2.COLOR_BGR2RGB)
   mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
   detector = vision.FaceLandmarker.create_from_options(
       vision.FaceLandmarkerOptions(base_options=python.BaseOptions(
           model_asset_path="face_landmarker.task"), num_faces=1))
   landmarks = np.array([[lm.x * image.shape[1], lm.y * image.shape[0]]
                         for lm in detector.detect(mp_image).face_landmarks[0]])

   # Triangulate and render
   tri = Delaunay(landmarks)
   output = np.zeros_like(image)
   for simplex in tri.simplices:
       pts = landmarks[simplex].astype(np.int32)
       centroid = pts.mean(axis=0).astype(int)
       color = image[centroid[1], centroid[0]].tolist()
       cv2.fillPoly(output, [pts], color)

The pipeline follows three key steps: detect facial landmarks, triangulate the points, and fill each triangle with sampled colors from the original image. The result is a geometric abstraction that preserves the essential features of the face while creating a distinctive artistic effect.

.. figure:: comparison_grid.png
   :width: 700px
   :align: center
   :alt: Side-by-side comparison of original portrait photograph and low-poly triangulated version

   Original portrait transformed into low-poly art using 478 facial landmarks and Delaunay triangulation.

.. note::

   The sample face image used in this exercise is from Unsplash (CC0 license).
   Download: https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800&q=80


Core Concepts
=============

Concept 1: Face Detection vs. Face Recognition
----------------------------------------------

**Face detection** and **face recognition** are often confused, but they serve fundamentally different purposes [ViolaJones2004]_:

* **Face Detection**: Locates faces in an image and provides their bounding boxes or landmark positions. It answers "Where are the faces?" without identifying who they belong to.

* **Face Recognition**: Identifies specific individuals by comparing detected faces against a database. It answers "Who is this person?"

This exercise focuses exclusively on face detection, specifically using **facial landmark detection** to find 478 precise points on a face. These landmarks enable creative applications without any identification or privacy concerns.

The evolution of face detection algorithms reflects decades of computer vision research:

* **1990s-2000s**: Haar Cascade classifiers using hand-crafted features [ViolaJones2004]_
* **2010s**: Deep neural network approaches with improved accuracy
* **2020s**: MediaPipe and similar frameworks enabling real-time detection on mobile devices [MediaPipe2019]_ [BlazeFace2019]_

.. admonition:: Did You Know?

   The Viola-Jones algorithm (2001) was revolutionary because it could detect faces at 15 frames per second on a 700 MHz processor, using a cascade of simple classifiers to quickly reject non-face regions [ViolaJones2004]_. This was the technology behind early digital cameras' face detection features.


Concept 2: The MediaPipe Face Mesh
----------------------------------

MediaPipe Face Mesh provides 478 three-dimensional landmarks covering the entire face surface [MediaPipe2019]_. Unlike simple bounding box detection, these landmarks capture detailed facial geometry including:

* **Face Oval**: 36 points defining the jawline and face contour
* **Eyes**: 128 points covering eyelids, corners, and surrounding area
* **Eyebrows**: 44 points for each eyebrow's shape
* **Nose**: 40 points from bridge to nostrils
* **Lips**: 40 points for inner and outer lip contours
* **Irises**: 10 points (5 per eye) for precise eye tracking
* **Face Surface**: 180 additional points covering cheeks and forehead

.. figure:: face_mesh_regions.png
   :width: 400px
   :align: center
   :alt: Schematic diagram showing color-coded facial regions with labeled landmark groups

   MediaPipe Face Mesh regions. Each color represents a different facial feature group, with 478 total landmarks providing detailed surface coverage. Diagram generated with Claude - Opus 4.5.

The MediaPipe Tasks API [MediaPipeDocs]_ provides a simple interface for detection:

.. code-block:: python
   :caption: Detecting facial landmarks
   :linenos:

   import mediapipe as mp
   from mediapipe.tasks import python
   from mediapipe.tasks.python import vision

   # Configure and create detector
   base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
   options = vision.FaceLandmarkerOptions(
       base_options=base_options,
       num_faces=1  # Detect up to 1 face
   )
   detector = vision.FaceLandmarker.create_from_options(options)

   # Detect landmarks in an image
   mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
   result = detector.detect(mp_image)

   # Extract pixel coordinates
   height, width = image_rgb.shape[:2]
   landmarks = []
   for lm in result.face_landmarks[0]:
       x = lm.x * width   # Convert normalized (0-1) to pixels
       y = lm.y * height
       landmarks.append((x, y))

Each landmark has normalized coordinates between 0 and 1, which must be multiplied by the image dimensions to get pixel positions.


Concept 3: From Landmarks to Low-Poly Art
-----------------------------------------

The connection between face detection and geometric art lies in Delaunay triangulation [Delaunay1934]_, covered in :ref:`Module 3.3.5 <module-3-3-5-delaunay>`. Facial landmarks provide semantically meaningful points that, when triangulated, create a mesh that follows facial features naturally.

The low-poly pipeline consists of four stages:

**Stage 1: Detect Landmarks**

MediaPipe provides 478 points distributed according to facial anatomy. Unlike random point placement, these points cluster around important features (eyes, nose, mouth) while remaining sparse on flat areas (cheeks, forehead).

**Stage 2: Add Boundary Points**

To ensure the triangulation covers the entire image (not just the face), add corner and edge points:

.. code-block:: python
   :caption: Adding boundary points for full coverage

   corners = np.array([
       [0, 0], [width, 0], [width, height], [0, height],
       [width/2, 0], [width, height/2], [width/2, height], [0, height/2]
   ])
   all_points = np.vstack([landmarks, corners])

**Stage 3: Triangulate**

Apply Delaunay triangulation [SciPyDocs]_ to connect all points into non-overlapping triangles:

.. code-block:: python
   :caption: Creating the triangle mesh

   from scipy.spatial import Delaunay
   triangulation = Delaunay(all_points)

**Stage 4: Sample and Render**

For each triangle, sample the color at its centroid and fill with that solid color using OpenCV [OpenCVDocs]_:

.. code-block:: python
   :caption: Rendering triangles with sampled colors

   for simplex in triangulation.simplices:
       triangle = all_points[simplex].astype(np.int32)
       centroid = np.mean(triangle, axis=0).astype(int)
       color = image[centroid[1], centroid[0]]  # Sample at centroid
       cv2.fillPoly(output, [triangle], color.tolist())

The result is a geometric abstraction where facial features remain recognizable because the triangle density is highest around eyes, nose, and mouth, exactly where humans focus their attention.

.. figure:: lowpoly_face_output.png
   :width: 500px
   :align: center
   :alt: Final low-poly face art showing triangulated portrait with color-sampled fills

   Low-poly face art generated from 478 landmarks plus 8 boundary points, creating approximately 960 triangles.


Hands-On Exercises
==================

Exercise 1: Execute and Explore
-------------------------------

Run the basic face detection script to visualize all 478 landmarks:

:download:`Download face_detection_basic.py <face_detection_basic.py>`

.. code-block:: bash

   python face_detection_basic.py

The script detects facial landmarks and draws them with color-coded regions:

* **Green**: Face oval (jawline)
* **Blue**: Eyebrows
* **Orange**: Eyes
* **Yellow**: Nose
* **Red**: Lips
* **Cyan**: Irises
* **Gray**: Face surface (cheeks, forehead)

.. figure:: face_detection_basic.png
   :width: 500px
   :align: center
   :alt: Face photograph with 478 colored landmark points overlaid showing different facial regions

   All 478 MediaPipe landmarks visualized with color-coded regions. Notice the higher density around eyes and lips.

After running the code, answer these reflection questions:

1. How many total landmarks does MediaPipe detect on a face?
2. Which facial features have the highest landmark density?
3. What happens when no face is detected in the image?

.. dropdown:: Answers and Explanation
   :class-title: sd-font-weight-bold

   1. **478 landmarks**: MediaPipe Face Mesh detects 478 3D landmarks (x, y, z coordinates, though we primarily use x and y).

   2. **Highest density regions**: Eyes and lips have the most landmarks because these features carry the most expressive information. The mouth area has approximately 40 landmarks for detailed lip tracking.

   3. **No face detected**: The ``result.face_landmarks`` list will be empty. Always check ``if result.face_landmarks:`` before accessing landmarks to avoid index errors.


Exercise 2: Modify Parameters
-----------------------------

Experiment with the face landmark visualization by modifying these aspects.

**Goal 1**: Filter landmarks to show only specific regions

Modify the visualization to show only eye landmarks (indices 33-133 and 263-364):

.. code-block:: python
   :caption: Filtering to show only eye regions

   for idx, landmark in enumerate(face_landmarks):
       # Only draw eye landmarks
       if (33 <= idx <= 133) or (263 <= idx <= 364):
           x = int(landmark.x * width)
           y = int(landmark.y * height)
           cv2.circle(annotated_image, (x, y), 3, (255, 128, 0), -1)

**Goal 2**: Change the landmark visualization style

Instead of colored circles, draw connected lines between landmarks:

.. code-block:: python
   :caption: Connecting landmarks with lines

   # Draw connections between consecutive landmarks
   points = [(int(lm.x * width), int(lm.y * height)) for lm in face_landmarks]
   for i in range(len(points) - 1):
       cv2.line(annotated_image, points[i], points[i+1], (0, 255, 0), 1)

**Goal 3**: Visualize landmark indices

Add text labels showing landmark numbers (useful for understanding the mesh topology):

.. code-block:: python
   :caption: Labeling key landmarks with indices

   key_indices = [1, 33, 133, 362, 263, 13, 14, 152]  # Nose, eyes, lips, chin
   for idx in key_indices:
       lm = face_landmarks[idx]
       x, y = int(lm.x * width), int(lm.y * height)
       cv2.putText(annotated_image, str(idx), (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


Exercise 3: Create Low-Poly Face Art
------------------------------------

Build the complete low-poly face transformation by completing the starter code below.

:download:`Download lowpoly_starter.py <lowpoly_starter.py>`

**Requirements**:

* Load an image and detect facial landmarks
* Apply Delaunay triangulation to the landmark points
* Add boundary points for full canvas coverage
* Sample colors from the original image at triangle centroids
* Render the result as filled triangles


.. dropdown:: Hint 1: Creating the detector
   :class-title: sd-font-weight-bold

   .. code-block:: python

      base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
      options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
      detector = vision.FaceLandmarker.create_from_options(options)
      result = detector.detect(mp_image)

.. dropdown:: Hint 2: Extracting landmarks
   :class-title: sd-font-weight-bold

   .. code-block:: python

      landmarks = []
      for lm in result.face_landmarks[0]:
          x = lm.x * width
          y = lm.y * height
          landmarks.append([x, y])
      landmarks = np.array(landmarks)

.. dropdown:: Complete Solution
   :class-title: sd-font-weight-bold

   See the complete implementation in ``lowpoly_face.py``.

   .. code-block:: python
      :linenos:

      import cv2
      import numpy as np
      from scipy.spatial import Delaunay
      import mediapipe as mp
      from mediapipe.tasks import python
      from mediapipe.tasks.python import vision

      # Load image
      image_bgr = cv2.imread("sample_face.jpg")
      image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
      height, width = image_rgb.shape[:2]

      # Create detector
      base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
      options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
      detector = vision.FaceLandmarker.create_from_options(options)

      # Detect landmarks
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
      result = detector.detect(mp_image)

      # Extract landmark coordinates
      landmarks = np.array([[lm.x * width, lm.y * height]
                            for lm in result.face_landmarks[0]])

      # Add boundary points
      corners = np.array([
          [0, 0], [width, 0], [width, height], [0, height],
          [width/2, 0], [width, height/2], [width/2, height], [0, height/2]
      ])
      all_points = np.vstack([landmarks, corners])

      # Triangulate
      triangulation = Delaunay(all_points)

      # Render
      output = np.zeros_like(image_rgb)
      for simplex in triangulation.simplices:
          triangle = all_points[simplex].astype(np.int32)
          centroid = np.mean(triangle, axis=0).astype(int)
          cx = np.clip(centroid[0], 0, width - 1)
          cy = np.clip(centroid[1], 0, height - 1)
          color = image_rgb[cy, cx].tolist()
          cv2.fillPoly(output, [triangle], color)

      cv2.imwrite("my_lowpoly_face.png", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
      detector.close()

**Challenge Extension**: Add white edge lines to create a "stained glass" effect:

.. code-block:: python
   :caption: Adding edge visualization

   # After filling all triangles, draw edges
   for simplex in triangulation.simplices:
       triangle = all_points[simplex].astype(np.int32)
       cv2.polylines(output, [triangle], True, (255, 255, 255), 1)

.. figure:: lowpoly_face_edges.png
   :width: 500px
   :align: center
   :alt: Low-poly face art with white triangle edges visible, creating a stained glass effect

   Low-poly face with visible edges, creating a stained glass aesthetic.


Exercise 4: Real-Time Webcam Low-Poly Face
------------------------------------------

Transform your live webcam feed into real-time low-poly art.

:download:`Download realtime_lowpoly.py <realtime_lowpoly.py>`

Run the real-time script:

.. code-block:: bash

   python realtime_lowpoly.py

**Controls**:

* **Q**: Quit the application
* **S**: Save the current frame as an image
* **E**: Toggle edge visibility
* **F**: Toggle FPS display

The script applies the same low-poly pipeline to each video frame, processing at approximately 15-30 FPS depending on your hardware. Key adaptations for real-time processing include:

1. **Video Mode**: MediaPipe uses ``RunningMode.VIDEO`` for temporal consistency
2. **Timestamp Handling**: Each frame requires a monotonically increasing timestamp
3. **Efficient Rendering**: OpenCV's ``fillPoly`` is faster than matplotlib for real-time use

.. code-block:: python
   :caption: Key differences for video processing
   :linenos:

   # Configure for video mode (not static images)
   options = vision.FaceLandmarkerOptions(
       base_options=base_options,
       running_mode=vision.RunningMode.VIDEO,  # Important!
       num_faces=1
   )

   # Each frame needs a timestamp (milliseconds)
   timestamp_ms = int(time.time() * 1000)
   result = detector.detect_for_video(mp_image, timestamp_ms)

**Challenge**: Modify the real-time script to support multiple faces by changing ``num_faces=1`` to ``num_faces=3`` and iterating over all detected faces.


TouchDesigner Extension (Optional)
==================================

:Duration: +20 minutes (optional)

For those with TouchDesigner experience, this extension demonstrates how the NumPy-based
face detection concepts translate directly to real-time performance applications.

.. note::

   **Requirements**:

   * TouchDesigner 2022.20000 or later (tested on 2025.31310)
   * Python 3.11 with ``scipy`` installed
   * MediaPipe TouchDesigner Plugin

**Download the complete project**:

:download:`Download TouchDesigner Project <face_detection_delaunay.toe>`

:download:`Download Script SOP Code <delaunay_script_sop.py>`

**Output Examples**:

.. list-table::
   :widths: 50 50
   :class: borderless

   * - .. figure:: face-delaunay-triangulation-gif.gif
          :width: 100%
          :alt: Wireframe Delaunay triangulation on face

          Wireframe triangulation effect

     - .. figure:: face-delaunay-triangulation-hero-gif.gif
          :width: 100%
          :alt: Delaunay triangulation with noise distortion

          With noise distortion effect

.. dropdown:: Step 1: Environment Setup
   :class-title: sd-font-weight-bold

   TouchDesigner requires external Python packages to be installed separately
   and configured in preferences.

   **Quick Setup**:

   1. Install Python 3.11 (must match TouchDesigner's Python version)
   2. Install scipy: ``pip install scipy numpy``
   3. In TouchDesigner: **Edit** > **Preferences** > **Python 64 bit Module Path**
   4. Set path to your Python site-packages folder

   **Verify installation** by creating a Text DAT and running:

   .. code-block:: python

      import scipy
      print(scipy.__version__)

   For detailed setup instructions, see the `TouchDesigner Python Documentation <https://docs.derivative.ca/Python>`_.

.. dropdown:: Step 2: MediaPipe Face Tracking Setup
   :class-title: sd-font-weight-bold

   Download and set up the MediaPipe plugin:

   1. Download from `GitHub: torinmb/mediapipe-touchdesigner <https://github.com/torinmb/mediapipe-touchdesigner>`_
   2. Drag ``MediaPipe.tox`` into your project
   3. Drag ``face_tracking.tox`` into your project
   4. Connect your webcam to the MediaPipe component

   **Network structure**:

   ::

      Video Device In TOP --> MediaPipe --> face_tracking

   The ``face_tracking`` component outputs a SOP with 478 face landmark points.

.. dropdown:: Step 3: Create Delaunay Script SOP
   :class-title: sd-font-weight-bold

   Create a Script SOP to triangulate the face landmarks:

   1. Press **Tab** > type "Script" > add **Script SOP**
   2. Connect the face_tracking SOP output to the Script SOP input
   3. In the Script SOP's callback DAT, paste this code:

   .. code-block:: python
      :linenos:
      :caption: delaunay_script_sop.py

      import numpy as np
      import scipy.spatial as sc

      def cook(scriptOp):
          scriptOp.clear()

          # Get input points (face landmarks)
          input_sop = scriptOp.inputs[0]
          if input_sop is None or len(input_sop.points) < 3:
              return

          # Extract 2D points for Delaunay triangulation
          points = [[p.x, p.y] for p in input_sop.points]

          # Compute Delaunay triangulation
          tri = sc.Delaunay(points)

          # Create triangular polygons
          for ia, ib, ic in tri.simplices:
              poly = scriptOp.appendPoly(3, closed=True)
              for idx, pt_idx in enumerate([ia, ib, ic]):
                  p = input_sop.points[pt_idx]
                  poly[idx].point.P = (p.x, p.y, p.z)

   **Key concepts**:

   * **Line 11**: Extract only X and Y for 2D triangulation
   * **Line 14**: ``scipy.spatial.Delaunay`` computes the triangulation
   * **Lines 17-21**: Create polygons using the triangle indices

.. dropdown:: Step 4: Render the Triangulated Mesh
   :class-title: sd-font-weight-bold

   Set up the rendering pipeline to visualize the result:

   **4.1 Create Geometry COMP**:

   1. Press **Tab** > add **Geometry** COMP
   2. Double-click to go inside, delete the default ``torus1``
   3. Add an **In SOP** inside
   4. Press **U** to go back up
   5. Connect ``script1`` output to ``geo1`` input

   **4.2 Add Wireframe Material**:

   1. Press **Tab** > add **Wireframe** MAT
   2. In ``geo1`` parameters > **Render** tab > **Material**: set to ``wireframe1``

   **4.3 Create Camera and Render**:

   1. Add **Camera** COMP (orthographic recommended for 2D face overlay)
   2. Add **Render** TOP
   3. In Render TOP: set **Camera** to ``camera1``, **Geometry** to ``geo1``

   **4.4 Composite over Webcam**:

   1. Add **Composite** TOP
   2. Connect webcam feed (from MediaPipe) to input 0
   3. Connect Render TOP to input 1
   4. Set **Operation** to "Over" or "Add"

   **Final network**:

   ::

      MediaPipe --> face_tracking --> script1 --> geo1 --> render1 --> composite1
          |                                         ^           ^
          |                                     camera1     [webcam]
          +-------------------------------------------------->

.. dropdown:: Step 5: Artistic Enhancements (Optional)
   :class-title: sd-font-weight-bold

   **Adding Noise for Organic Movement**:

   Add a Noise SOP after the Script SOP to create organic, flowing distortion:

   1. Press **Tab** > add **Noise** SOP
   2. Connect ``script1`` output to ``noise1`` input
   3. Connect ``noise1`` output to ``geo1`` input (instead of script1)

   **Recommended Noise Parameters**:

   * **Type**: Sparse
   * **Amplitude**: 0.02 - 0.05 (subtle movement)
   * **Period**: 2 - 4
   * **Roughness**: 0.5
   * **Translate Z**: Use an LFO or ``absTime.seconds * 0.5`` for animation

   The noise creates a subtle organic ripple effect across the triangulated mesh,
   making the visualization feel more alive and dynamic.

   **Neon Glow Effect**:

   1. Add **Bloom** TOP after the Render TOP
   2. Increase **Size** and **Threshold** parameters
   3. Use a bright wireframe color (cyan, magenta, or white)

   **Audio Reactivity** (Advanced):

   1. Add **Audio Device In** CHOP
   2. Use **Analyze** CHOP to extract amplitude
   3. Drive wireframe thickness or noise intensity with audio levels

This extension demonstrates how creative coding concepts transfer from Python prototyping to real-time interactive installations.


Summary
=======

Key Takeaways
-------------

* **Face detection locates faces** without identifying who they belong to; face recognition identifies individuals
* **MediaPipe Face Mesh** provides 478 3D landmarks covering the entire face surface in real-time
* **Delaunay triangulation** applied to facial landmarks creates low-poly art that preserves facial features
* **Color sampling at centroids** produces the characteristic flat-shaded geometric look
* The technique **transfers directly** from still images to real-time video with minor API changes

Common Pitfalls
---------------

* **BGR vs RGB**: OpenCV loads images as BGR, but MediaPipe expects RGB. Always convert with ``cv2.cvtColor(image, cv2.COLOR_BGR2RGB)``
* **Normalized coordinates**: MediaPipe landmarks are 0-1 normalized. Multiply by image width/height to get pixels
* **Empty results**: Always check ``if result.face_landmarks:`` before accessing landmarks
* **Centroid bounds**: When sampling colors, clip centroid coordinates to valid image bounds to avoid index errors
* **Video timestamps**: In video mode, timestamps must be monotonically increasing (never go backward)


References
==========

.. [ViolaJones2004] Viola, P., & Jones, M. J. (2004). Robust Real-Time Face Detection. *International Journal of Computer Vision*, 57(2), 137-154. https://doi.org/10.1023/B:VISI.0000013087.49260.fb

.. [MediaPipe2019] Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M., ... & Grundmann, M. (2019). MediaPipe: A Framework for Building Perception Pipelines. *arXiv preprint*. https://arxiv.org/abs/1906.08172

.. [BlazeFace2019] Bazarevsky, V., Kartynnik, Y., Vakunov, A., Raveendran, K., & Grundmann, M. (2019). BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs. *arXiv preprint*. https://arxiv.org/abs/1907.05047

.. [Delaunay1934] Delaunay, B. (1934). Sur la sphere vide. *Bulletin de l'Academie des Sciences de l'URSS*, 6, 793-800.

.. [MediaPipeDocs] Google. (2024). Face Landmarker guide for Python. *Google AI Edge*. https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python

.. [OpenCVDocs] OpenCV Developers. (2024). OpenCV Python Tutorials. *OpenCV Documentation*. https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html

.. [SciPyDocs] SciPy Developers. (2024). scipy.spatial.Delaunay. *SciPy Documentation*. https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html

.. [MediaPipeTD] Blankensmith, T. (2024). MediaPipe TouchDesigner Plugin. *GitHub*. https://github.com/torinmb/mediapipe-touchdesigner

