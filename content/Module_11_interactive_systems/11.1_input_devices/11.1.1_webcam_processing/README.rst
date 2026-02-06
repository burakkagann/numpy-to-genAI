.. _module-11-1-1-webcam-processing:

=====================================
11.1.1 - Webcam Processing
=====================================

:Duration: 35 minutes
:Level: Intermediate
:Prerequisites: Module 1 (Image Arrays), Module 3 (Transformations)

.. contents:: Contents
   :local:
   :depth: 2

Overview
========

Your webcam is more than a video chat tool - it is a real-time image generator producing 30 frames per second, each frame a NumPy array ready for creative manipulation. In this exercise, you will learn to capture this stream and transform it into interactive visual art.

Webcam processing forms the foundation of interactive installations, motion-reactive visuals, and computer vision applications. By the end of this module, you will understand how to capture frames, apply real-time effects, and detect motion - skills that connect directly to generative art and AI-powered creative systems.

**Learning Objectives**

By completing this exercise, you will:

* Capture live video frames from a webcam using OpenCV's VideoCapture API
* Apply real-time image processing filters to video streams
* Implement background subtraction to detect motion
* Understand the capture-process-display pipeline fundamental to interactive systems


Quick Start: See Your Webcam as Data
====================================

Let's immediately see what webcam capture looks like. Run this minimal example:

.. code-block:: python
   :caption: Minimal webcam capture
   :linenos:

   import cv2

   cap = cv2.VideoCapture(0)  # Open default webcam

   while True:
       ret, frame = cap.read()  # Read one frame
       if not ret:
           break
       cv2.imshow('Webcam', frame)  # Display it
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break

   cap.release()
   cv2.destroyAllWindows()

Press 'q' to quit. Each frame you see is a NumPy array with shape ``(height, width, 3)`` - the same data structure you have been working with throughout this course.

.. figure:: webcam_frame.png
   :width: 400px
   :align: center
   :alt: Sample webcam frame showing typical video capture output

   A webcam frame is simply a NumPy array. Notice the resolution (typically 640x480 or higher) and the three color channels (BGR in OpenCV).

.. tip::

   OpenCV uses BGR color order, not RGB. This matters when combining OpenCV code with other libraries like PIL or matplotlib that expect RGB.


Video Capture Fundamentals
==========================

How Webcams Generate Data
-------------------------

A webcam continuously captures light through its sensor and converts it to digital data. This happens in a loop:

1. **Sensor captures light** - The camera sensor reads incoming photons
2. **Analog-to-digital conversion** - Light intensity becomes pixel values
3. **Frame assembly** - Pixels form a complete image (one frame)
4. **Transfer to computer** - The frame arrives as a NumPy array

This cycle repeats 24-60 times per second (frames per second, or FPS). Your code reads these frames one at a time.

.. code-block:: python

   # The frame is a NumPy array
   ret, frame = cap.read()
   print(f"Frame shape: {frame.shape}")  # e.g., (480, 640, 3)
   print(f"Data type: {frame.dtype}")    # uint8 (values 0-255)

.. important::

   Always check the ``ret`` value. It returns ``False`` if the camera fails to capture a frame (disconnected, busy, or end of video file).

The VideoCapture Object
-----------------------

OpenCV's ``VideoCapture`` class handles the connection to your webcam:

.. code-block:: python

   import cv2

   # Open webcam (0 = default camera, 1 = second camera, etc.)
   cap = cv2.VideoCapture(0)

   # Check if opened successfully
   if not cap.isOpened():
       print("Cannot open camera")
       exit()

   # Read frames in a loop
   while True:
       ret, frame = cap.read()
       if not ret:
           break
       # ... process frame ...

   # Always release when done
   cap.release()

The pattern is always: **open, read in loop, release**. Forgetting to release can leave your camera locked.

.. admonition:: Did You Know?

   The same ``VideoCapture`` class works with video files. Replace ``0`` with a filename like ``"video.mp4"`` to process recorded video frame-by-frame (Bradski & Kaehler, 2008).


Real-Time Frame Processing
==========================

The Processing Pipeline
-----------------------

Interactive video applications follow a consistent pattern:

.. figure:: processing_pipeline.png
   :width: 600px
   :align: center
   :alt: Diagram showing capture, process, display pipeline

   The webcam processing pipeline: Capture a frame, transform it, display the result. This loop runs continuously at video frame rate.

Each iteration of your main loop performs these three steps. The "process" step is where your creativity comes in - any image transformation you have learned can be applied here.

Applying Filters in Real-Time
-----------------------------

Since each frame is a NumPy array, you can apply any image operation:

.. code-block:: python

   while True:
       ret, frame = cap.read()
       if not ret:
           break

       # Convert to grayscale
       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

       # Apply Gaussian blur
       blurred = cv2.GaussianBlur(frame, (21, 21), 0)

       # Detect edges
       edges = cv2.Canny(gray, 50, 150)

       cv2.imshow('Result', edges)
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break

.. figure:: effects_comparison.png
   :width: 400px
   :align: center
   :alt: Grid showing original, grayscale, blur, and edge detection effects

   Common real-time effects: original, grayscale, blur, and edge detection. Each transformation runs fast enough for 30+ FPS processing.

.. note::

   The ``waitKey(1)`` call is essential. It waits 1 millisecond for a keypress and also allows OpenCV to update the display window. Without it, no image appears.


Background Subtraction
======================

Detecting What Moves
--------------------

Background subtraction identifies moving objects by comparing frames. The core idea is simple: if a pixel changes significantly between frames, something moved there.

**Frame Differencing** compares consecutive frames:

.. code-block:: python

   # Store previous frame
   ret, previous = cap.read()
   previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)

   while True:
       ret, current = cap.read()
       current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)

       # Calculate absolute difference
       diff = cv2.absdiff(previous_gray, current_gray)

       # Threshold to create binary mask
       _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

       # Update previous for next iteration
       previous_gray = current_gray.copy()

.. figure:: motion_detection.png
   :width: 700px
   :align: center
   :alt: Four panels showing frame differencing: Frame t, Frame t+1, Difference mask, Detection result

   Frame differencing in action: comparing two frames reveals where movement occurred. The white areas in the difference mask indicate motion.

Creating Motion Visualizations
------------------------------

Once you have a motion mask, you can create visual effects:

.. code-block:: python

   # Highlight motion in green
   output = current.copy()
   output[motion_mask > 0] = [0, 255, 0]  # Green where motion

   # Or blend for softer effect
   overlay = current.copy()
   overlay[motion_mask > 0] = [0, 255, 0]
   output = cv2.addWeighted(current, 0.7, overlay, 0.3, 0)

.. admonition:: Did You Know?

   Background subtraction has been studied extensively in computer vision. Piccardi (2004) provides a comprehensive survey of techniques ranging from simple frame differencing to sophisticated statistical models [Piccardi2004]_. OpenCV provides built-in background subtractor classes for more advanced applications [IntelOpenCV]_.


Hands-On Exercises
==================

These exercises follow the Execute-Modify-Create progression. Start by running existing code, then modify it, then build your own.

Exercise 1: Execute and Explore
-------------------------------

**Time estimate:** 4 minutes

Run the basic webcam capture script to observe how frames are captured:

.. code-block:: python
   :caption: webcam_capture.py
   :linenos:

   import cv2

   cap = cv2.VideoCapture(0)

   if not cap.isOpened():
       print("Error: Could not open webcam")
       exit()

   print("Press 'q' to quit, 's' to save a frame")

   while True:
       ret, frame = cap.read()
       if not ret:
           break

       cv2.imshow('Webcam Feed', frame)

       key = cv2.waitKey(1) & 0xFF
       if key == ord('q'):
           break
       elif key == ord('s'):
           cv2.imwrite('saved_frame.png', frame)
           print(f"Saved! Shape: {frame.shape}")

   cap.release()
   cv2.destroyAllWindows()

**Reflection questions:**

1. What resolution does your webcam capture at? (Check the frame shape)
2. Press 's' to save a frame. Open it in an image viewer - does it look the same as what you saw on screen?
3. Why might the saved image have slightly different colors than expected?

.. dropdown:: Solution & Explanation

   **Answers:**

   1. Common resolutions are 640x480, 1280x720, or 1920x1080. Your frame shape shows ``(height, width, 3)``.

   2. The saved image should look identical since it is the raw frame data.

   3. OpenCV saves in BGR format. Some image viewers may interpret it as RGB, causing a blue/red color swap. This is the BGR vs RGB issue mentioned earlier.

Exercise 2: Modify to Add Effects
---------------------------------

**Time estimate:** 5 minutes

Using ``webcam_effects.py`` as a starting point, modify the code to achieve these goals:

**Goals:**

1. Add a "sepia" filter (warm, vintage look)
2. Create a "pixelate" effect by downscaling then upscaling
3. Add a mirror effect (flip horizontally)

.. dropdown:: Hints

   * Sepia uses a color transformation matrix applied with ``cv2.transform()``
   * Pixelate: use ``cv2.resize()`` to shrink, then resize back to original
   * Mirror: use ``cv2.flip(frame, 1)`` where 1 means horizontal flip

.. dropdown:: Solutions

   **1. Sepia filter:**

   .. code-block:: python

      # Sepia transformation matrix
      sepia_matrix = np.array([[0.272, 0.534, 0.131],
                               [0.349, 0.686, 0.168],
                               [0.393, 0.769, 0.189]])
      sepia = cv2.transform(frame, sepia_matrix)
      sepia = np.clip(sepia, 0, 255).astype(np.uint8)

   **2. Pixelate effect:**

   .. code-block:: python

      # Shrink to small size, then expand back
      small = cv2.resize(frame, (64, 48))
      pixelated = cv2.resize(small, (frame.shape[1], frame.shape[0]),
                             interpolation=cv2.INTER_NEAREST)

   **3. Mirror effect:**

   .. code-block:: python

      mirrored = cv2.flip(frame, 1)  # 1 = horizontal flip

Exercise 3: Create Your Own Motion Detector
-------------------------------------------

**Time estimate:** 6 minutes

Build a simple motion detector from scratch using frame differencing.

**Goal:** Create a script that highlights moving areas in the webcam feed.

**Requirements:**

* Compare current frame to previous frame
* Create a binary motion mask using thresholding
* Display the motion as a colored overlay

**Hints:**

* Convert frames to grayscale before comparing (reduces noise)
* Use ``cv2.absdiff()`` to find differences
* Use ``cv2.threshold()`` to create binary mask
* Apply the mask to colorize moving regions

.. code-block:: python
   :caption: Starter code (webcam_starter.py)

   import cv2
   import numpy as np

   cap = cv2.VideoCapture(0)

   ret, previous_frame = cap.read()
   previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

   while True:
       ret, frame = cap.read()
       if not ret:
           break

       current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

       # TODO: Calculate difference between current and previous
       diff = None  # Your code here

       # TODO: Create binary motion mask
       motion_mask = None  # Your code here

       # Display
       cv2.imshow('Webcam', frame)
       if diff is not None:
           cv2.imshow('Motion', motion_mask)

       previous_gray = current_gray.copy()

       if cv2.waitKey(1) & 0xFF == ord('q'):
           break

   cap.release()
   cv2.destroyAllWindows()

.. dropdown:: Complete Solution

   .. code-block:: python
      :caption: Motion detector solution
      :linenos:
      :emphasize-lines: 16-17, 20-21, 24-26

      import cv2
      import numpy as np

      cap = cv2.VideoCapture(0)

      ret, previous_frame = cap.read()
      previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
      previous_gray = cv2.GaussianBlur(previous_gray, (21, 21), 0)

      while True:
          ret, frame = cap.read()
          if not ret:
              break

          current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
          current_gray = cv2.GaussianBlur(current_gray, (21, 21), 0)

          # Calculate absolute difference
          diff = cv2.absdiff(previous_gray, current_gray)

          # Threshold to binary mask
          _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

          # Create colored overlay
          output = frame.copy()
          output[motion_mask > 0] = [0, 255, 0]  # Green on motion

          cv2.imshow('Motion Detector', output)

          previous_gray = current_gray.copy()

          if cv2.waitKey(1) & 0xFF == ord('q'):
              break

      cap.release()
      cv2.destroyAllWindows()

   **How it works:**

   * Lines 16-17: Blur reduces noise that would cause false motion detection
   * Lines 20-21: ``absdiff`` and ``threshold`` create the motion mask
   * Lines 24-26: The mask selects pixels to colorize, creating the visual effect

   **Challenge extension:** Add a motion trail by accumulating masks over several frames, or trigger a sound when motion exceeds a threshold.


Summary
=======

In 35 minutes, you have learned to transform your webcam from a passive recording device into an interactive input for generative art:

**Key takeaways:**

* Webcam frames are NumPy arrays - the same data structure used throughout this course
* The capture-process-display loop is the foundation of all interactive video applications
* Background subtraction detects motion by comparing frames over time
* Real-time processing requires efficient code that runs at 30+ FPS

**Common pitfalls to avoid:**

* Forgetting to release the camera (``cap.release()``) locks the device
* Missing the BGR to RGB conversion when using OpenCV with other libraries
* Not checking the ``ret`` value can cause crashes when the camera disconnects

These webcam processing skills prepare you for advanced topics like computer vision in TouchDesigner (Module 11.2), optical flow, and AI-powered pose detection.


Next Steps
==========

Continue to :doc:`../11.1.2_audio_reactivity/README` to learn how to process audio input for sound-reactive visuals, or explore :doc:`../../11.2_computer_vision_td/11.2.1_motion_detection/README` to see these techniques in TouchDesigner.


References
==========

.. [Bradski2008] Bradski, G., & Kaehler, A. (2008). *Learning OpenCV: Computer Vision with the OpenCV Library*. O'Reilly Media. ISBN: 978-0-596-51613-0. [Foundational text on OpenCV, covers VideoCapture in depth]

.. [OpenCVDocs] OpenCV Development Team. (2024). "VideoCapture Class Reference." *OpenCV Documentation*. https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html [Official API reference]

.. [Szeliski2022] Szeliski, R. (2022). *Computer Vision: Algorithms and Applications* (2nd ed.). Springer. https://szeliski.org/Book/ [Comprehensive academic reference]

.. [Piccardi2004] Piccardi, M. (2004). "Background subtraction techniques: a review." *IEEE International Conference on Systems, Man and Cybernetics*, 4, 3099-3104. [Survey of background subtraction methods]

.. [GonzalezWoods2018] Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing* (4th ed.). Pearson. ISBN: 978-0-13-335672-4. [Standard image processing textbook]

.. [NumPyDocs] NumPy Developers. (2024). "NumPy Reference." *NumPy Documentation*. https://numpy.org/doc/stable/reference/ [Array operations reference]

.. [IntelOpenCV] Intel Corporation. (2024). "Background Subtraction." *OpenCV Tutorials*. https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html [OpenCV background subtraction guide]
