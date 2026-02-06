.. _13.1.1_mediapipe_integration:

======================================
13.1.1 - MediaPipe Integration
======================================

:Duration: 45-50 minutes
:Level: Intermediate-Advanced
:Prerequisites: Module 11.2.3 (Face Detection), Module 10.1 (TD Environment)

Overview
========

In Module 11.2.3, you learned to detect 478 facial landmarks using MediaPipe Face Mesh. This module expands that foundation to full-body sensing: 33 pose landmarks, 21 hand landmarks per hand, object detection with bounding boxes, and person segmentation with pixel masks. The coordinate handling and real-time patterns you learned transfer directly---we will build on them.

MediaPipe is a cross-platform framework developed by Google that provides production-ready ML solutions for common perception tasks [Lugaresi2019]_. Unlike traditional computer vision pipelines requiring manual feature engineering, MediaPipe provides pre-trained neural networks that work out-of-the-box. For interactive artists and creative coders, this means real-time human sensing becomes accessible without deep ML expertise.

This module demonstrates how to integrate MediaPipe with TouchDesigner [Derivative2024]_ using OSC (Open Sound Control), creating a complete pipeline from webcam input to real-time visual effects. Building on research in human pose estimation [Cao2019]_ and natural user interfaces [Wigdor2011]_, we create systems where human movement drives interactive art [Edmonds2011]_.

Learning Objectives
-------------------

By the end of this exercise, you will be able to:

1. Apply MediaPipe knowledge from face detection to pose, hand, object detection, and segmentation (knowledge transfer)
2. Configure multi-solution detection pipelines for combined human sensing
3. Extract and route landmark data to TouchDesigner parameters via OSC in real-time
4. Optimize inference performance for interactive installations

Quick Start: See It In Action
=============================

First, download the required MediaPipe models by running:

.. code-block:: bash

   pip install python-osc
   python model_downloader.py

Then run the demo script to see all MediaPipe solutions in action:

.. code-block:: python
   :caption: mediapipe_solutions_demo.py (excerpt)

   import mediapipe as mp
   from mediapipe.tasks import python
   from mediapipe.tasks.python import vision

   # Same pattern as Face Landmarker in Module 11.2.3
   # BaseOptions -> [Solution]Options -> create_from_options

   # Pose detection (33 body landmarks)
   pose_options = vision.PoseLandmarkerOptions(
       base_options=python.BaseOptions(model_asset_path='pose_landmarker.task'),
       running_mode=vision.RunningMode.IMAGE,
       num_poses=1
   )
   pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

   # Run detection
   mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
   result = pose_detector.detect(mp_image)

.. figure:: mediapipe_solutions_overview.png
   :width: 700px
   :align: center
   :alt: Four-panel visualization showing pose skeleton, hand landmarks, object bounding boxes, and person segmentation

   MediaPipe Solutions in action. Top-left: Pose detection (33 body landmarks). Top-right: Hand detection (21 landmarks per hand). Bottom-left: Object detection (bounding boxes). Bottom-right: Segmentation (person mask).

Core Concepts
=============

Concept 1: MediaPipe Solutions Ecosystem
----------------------------------------

MediaPipe provides a unified API across multiple perception solutions. The pattern you learned in Module 11.2.3 for Face Landmarker applies to all solutions:

.. code-block:: text

   BaseOptions (model path) -> [Solution]Options (configuration) -> create_from_options

This consistency means that once you understand one solution, learning others is straightforward [MediaPipe2024]_. The key difference is what each solution outputs:

**Landmark-based solutions** (Face, Pose, Hand):
   - Return normalized (0-1) coordinates for each point
   - Include visibility or presence scores
   - Map naturally to CHOP channels in TouchDesigner

**Detection-based solutions** (Object Detector):
   - Return bounding boxes (x, y, width, height) using efficient architectures [Tan2020]_
   - Include class labels and confidence scores
   - Map naturally to DAT tables in TouchDesigner

**Segmentation solutions** (Image Segmenter):
   - Return pixel-level masks (numpy arrays)
   - Each pixel has a category value (person, background)
   - Map naturally to TOP textures in TouchDesigner

.. figure:: pose_landmarks_labeled.png
   :width: 600px
   :align: center
   :alt: Diagram of human body with 33 pose landmarks labeled and color-coded by region

   MediaPipe Pose provides 33 landmarks covering the face, arms, torso, and legs. Key landmarks for interactive applications include wrists (15, 16) for hand tracking and shoulders (11, 12) for body orientation.

.. figure:: hand_landmarks_labeled.png
   :width: 500px
   :align: center
   :alt: Diagram of hand with 21 landmarks labeled for each finger

   MediaPipe Hand provides 21 landmarks per hand [Zhang2020]_. The wrist (0) serves as the anchor point, with each finger having 4 landmarks from base to tip.

.. admonition:: Did You Know?

   MediaPipe uses lightweight neural networks optimized for mobile devices [Bazarevsky2020]_. The Pose model (BlazePose) runs at 30+ FPS on a typical smartphone, making it suitable for real-time interactive applications and art installations.


Concept 2: Data Types and TouchDesigner Mapping
-----------------------------------------------

Different MediaPipe solutions produce different data types, and each maps to a specific TouchDesigner operator family:

.. figure:: data_types_td_mapping.png
   :width: 700px
   :align: center
   :alt: Flowchart showing landmarks mapping to CHOP, bounding boxes to DAT, and masks to TOP

   Data type mapping from MediaPipe to TouchDesigner. The OSC protocol bridges the Python detection scripts with TD's operator network.

**Landmarks to CHOP** (Channel Operator):

Pose and hand landmarks are streams of floating-point values---perfect for CHOP channels. Each landmark produces 3-4 values (x, y, z, visibility), so 33 pose landmarks create 132 channels:

.. code-block:: python

   # Sending pose landmarks via OSC
   for i, lm in enumerate(pose_landmarks):
       osc_client.send_message(f"/pose/{i}", [lm.x, lm.y, lm.z, lm.visibility])

In TouchDesigner, an OSC In CHOP receives these as channels that can drive parameters:

.. code-block:: text

   OSC In CHOP (port 7000)
     └─ Select CHOP (pattern: /pose/*)
          └─ Math CHOP (scale to screen coordinates)
               └─ Parameter Export (to geometry position)

**Bounding Boxes to DAT** (Data Operator):

Object detections include class names and coordinates---tabular data ideal for DAT:

.. code-block:: python

   # Sending object detections via OSC
   osc_client.send_message("/objects/count", len(detections))
   for i, det in enumerate(detections):
       box = det.bounding_box
       osc_client.send_message(f"/object/{i}", [
           det.categories[0].category_name,
           det.categories[0].score,
           box.origin_x, box.origin_y, box.width, box.height
       ])

**Segmentation Masks to TOP** (Texture Operator):

Segmentation produces pixel arrays that can be sent as textures or used to control compositing:

.. code-block:: python

   # Extracting segmentation mask
   seg_result = segmenter.segment(mp_image)
   mask = seg_result.category_mask.numpy_view()  # HxW numpy array
   person_ratio = np.sum(mask > 0) / mask.size   # How much of frame is person


Concept 3: Real-Time OSC Pipeline
---------------------------------

OSC (Open Sound Control) is an industry-standard protocol for real-time communication between creative applications [Wright2005]_. It enables low-latency data transfer between Python (running MediaPipe) and TouchDesigner.

.. figure:: osc_pipeline_architecture.png
   :width: 700px
   :align: center
   :alt: Pipeline diagram showing webcam to Python/MediaPipe to OSC to TouchDesigner to visual output

   Complete pipeline architecture. The Python script captures video, runs multiple MediaPipe detectors, and sends results via OSC. TouchDesigner receives OSC messages and routes them to visual effects.

**Setting Up the OSC Connection:**

.. code-block:: python

   from pythonosc import udp_client

   # Create OSC client (TouchDesigner listens on port 7000)
   osc_client = udp_client.SimpleUDPClient("127.0.0.1", 7000)

   # Send data (same timestamp for all detectors ensures sync)
   timestamp_ms = int(time.time() * 1000)

   # Run all detections
   pose_result = pose_detector.detect_for_video(mp_image, timestamp_ms)
   hand_result = hand_detector.detect_for_video(mp_image, timestamp_ms)
   obj_result = object_detector.detect_for_video(mp_image, timestamp_ms)

   # Send results via OSC
   if pose_result.pose_landmarks:
       for i, lm in enumerate(pose_result.pose_landmarks[0]):
           osc_client.send_message(f"/pose/{i}", [lm.x, lm.y, lm.z, lm.visibility])

**OSC Address Patterns:**

The hierarchical address structure makes it easy to filter data in TouchDesigner:

.. code-block:: text

   /pose/{0-32}              - 33 body landmarks [x, y, z, visibility]
   /hand/left/{0-20}         - Left hand landmarks [x, y, z]
   /hand/right/{0-20}        - Right hand landmarks [x, y, z]
   /object/{0-n}             - Object detections [class, conf, x, y, w, h]
   /objects/count            - Number of detected objects
   /mask/person_ratio        - Ratio of frame containing person (0-1)
   /status/fps               - Current processing FPS

.. admonition:: Important

   VIDEO mode requires timestamps for temporal consistency. Always use the same timestamp for all detectors processing the same frame---this ensures synchronized data output.


Hands-On Exercises
==================

Exercise 1: Execute and Explore
-------------------------------

Run the OSC sender script and observe data flowing to TouchDesigner.

**Setup:**

1. Ensure models are downloaded: ``python model_downloader.py``
2. Install python-osc: ``pip install python-osc``
3. Start TouchDesigner with an OSC In CHOP on port 7000

**Run the script:**

.. code-block:: bash

   python mediapipe_osc_sender.py

**Observation Tasks:**

1. Move in front of the webcam and watch the pose skeleton overlay
2. Raise your hands to see hand landmark detection activate
3. In TouchDesigner, observe the OSC In CHOP receiving channels

**Reflection Questions:**

1. How does the number of pose landmarks (33) compare to face landmarks (478) you learned in Module 11.2.3?
2. What is the difference between ``visibility`` (pose) and ``presence`` (face)?
3. Which OSC address patterns appear when you raise only your left hand?
4. How would you filter only left hand data in TouchDesigner?

.. dropdown:: Answers and Explanation
   :class-title: sd-font-weight-bold

   1. **Landmark count comparison**: Face Mesh has 478 landmarks for detailed facial geometry (needed for expressions and AR filters), while Pose has 33 landmarks focused on major body joints (sufficient for gesture and movement tracking). The design principle from Module 11.2.3 applies: more landmarks where more precision is needed.

   2. **Visibility vs Presence**: Visibility (pose) indicates whether a joint is occluded (behind the body or off-screen)---useful for knowing if tracking is reliable. Presence (face) indicates detection confidence. Both are 0-1 scales but serve different purposes.

   3. **Left hand addresses**: When raising only the left hand, you will see ``/hand/left/0`` through ``/hand/left/20`` appearing in the OSC In CHOP.

   4. **Filtering in TD**: Use a Select CHOP with pattern ``/hand/left/*`` to isolate left hand data. The pattern ``/pose/15`` would give you just the left wrist (useful for tracking one hand).


Exercise 2: Modify Parameters - Particle Control
-------------------------------------------------

Run the particle sender script and drive visual effects from hand position.

**Run the script:**

.. code-block:: bash

   python mediapipe_particles_sender.py

This script sends simplified control data:

- ``/particles/left_hand [x, y]`` - Left hand normalized position
- ``/particles/right_hand [x, y]`` - Right hand normalized position
- ``/particles/arm_spread [value]`` - Distance between hands (0-1)
- ``/particles/body_center [x, y]`` - Center of body
- ``/particles/active [0 or 1]`` - Whether tracking is active

**Goal 1**: Configure OSC address mapping

Modify the Python script to use different OSC addresses. For example, change ``/particles/`` to ``/control/`` and verify TouchDesigner still receives the data with the new pattern.

.. dropdown:: Hint
   :class-title: sd-font-weight-bold

   Search for all ``osc_client.send_message`` calls and update the address string prefix.

**Goal 2**: Add derived measurements

Add a new OSC message that sends the vertical position of the body center (useful for jump detection):

.. dropdown:: Solution
   :class-title: sd-font-weight-bold

   In ``extract_particle_controls()``, add:

   .. code-block:: python

      # Add vertical position (0=top, 1=bottom)
      return {
          ...
          'vertical': body_center_y,  # New
      }

   In ``send_particle_controls()``, add:

   .. code-block:: python

      osc_client.send_message("/particles/vertical", controls['vertical'])

**Goal 3**: Create visual effect mapping

In TouchDesigner, map the arm spread value to a particle system property:

- Arm spread (0-1) -> Particle birth rate (0-1000)
- Hand positions -> Particle emitter positions

This creates an interaction where spreading your arms spawns more particles.


Exercise 3: Build Complete OSC Bridge
-------------------------------------

Complete the starter code to create a full multi-model detection pipeline.

**Download and examine:**

:download:`td_osc_bridge_starter.py <td_osc_bridge_starter.py>`

**Your Tasks:**

1. Initialize all four detectors (pose, hand, object, segmentation)
2. Set up OSC client on port 7000
3. Process webcam frames in real-time
4. Send pose landmarks via ``/pose/{joint_id}``
5. Send hand landmarks via ``/hand/{left|right}/{joint_id}``
6. Send object detections via ``/object/{id}``
7. (Bonus) Send segmentation info via ``/mask/person_ratio``

.. dropdown:: Hint 1: Model Loading Pattern
   :class-title: sd-font-weight-bold

   Use the same pattern from Module 11.2.3:

   .. code-block:: python

      def create_pose_detector():
          model_path = setup_model_path("pose_landmarker.task")
          base_options = python.BaseOptions(model_asset_path=model_path)
          options = vision.PoseLandmarkerOptions(
              base_options=base_options,
              running_mode=vision.RunningMode.VIDEO,
              num_poses=1
          )
          return vision.PoseLandmarker.create_from_options(options)

.. dropdown:: Hint 2: OSC Sending
   :class-title: sd-font-weight-bold

   Loop through landmarks and send each with its index:

   .. code-block:: python

      def send_pose_data(osc_client, pose_landmarks):
          for i, lm in enumerate(pose_landmarks):
              osc_client.send_message(f"/pose/{i}", [lm.x, lm.y, lm.z, lm.visibility])

.. dropdown:: Hint 3: Main Loop Structure
   :class-title: sd-font-weight-bold

   Process each frame with all detectors:

   .. code-block:: python

      # Get timestamp (same for all detectors)
      timestamp_ms = int(time.time() * 1000)

      # Run detections
      pose_result = pose_detector.detect_for_video(mp_image, timestamp_ms)
      hand_result = hand_detector.detect_for_video(mp_image, timestamp_ms)

      # Send data
      if pose_result.pose_landmarks:
          send_pose_data(osc_client, pose_result.pose_landmarks[0])

.. dropdown:: Complete Solution
   :class-title: sd-font-weight-bold

   :download:`td_osc_bridge_solution.py <td_osc_bridge_solution.py>`

**Challenge Extension:**

Add face landmarks from Module 11.2.3 to create holistic tracking. You will need to:

1. Add the face landmarker model to ``model_downloader.py``
2. Create a face detector using the pattern from 11.2.3
3. Send face landmarks via ``/face/{id}`` addresses

This demonstrates the knowledge transfer from Module 11.2.3---the API pattern is identical.


TouchDesigner Network Reference
===============================

Each exercise includes corresponding TouchDesigner companion files. The basic network structure for receiving MediaPipe data:

.. code-block:: text

   [OSC In CHOP]
        │
        ├──► [Select CHOP] pattern: /pose/*
        │         │
        │         └──► [Math CHOP] scale/offset
        │                   │
        │                   └──► [Parameter Export]
        │
        ├──► [Select CHOP] pattern: /hand/left/*
        │         │
        │         └──► [Geometry COMP] drive position
        │
        └──► [Select CHOP] pattern: /objects/count
                  │
                  └──► [DAT Execute] trigger events

**Key TouchDesigner Settings:**

- OSC In CHOP: Port 7000, Network Protocol: UDP
- Select CHOP: Use patterns like ``/pose/*`` or ``/hand/left/*``
- Math CHOP: Scale from 0-1 to screen coordinates (e.g., multiply by resolution)


Summary
=======

Key Takeaways
-------------

1. **API Pattern Transfer**: The MediaPipe Tasks API pattern (BaseOptions -> Options -> create_from_options) applies consistently across all solutions---knowledge from face detection transfers directly to pose, hand, object, and segmentation.

2. **Data Type Mapping**: Different MediaPipe outputs map to specific TD operator types:

   - Landmarks (continuous values) -> CHOP
   - Bounding boxes (tabular data) -> DAT
   - Segmentation masks (pixel arrays) -> TOP

3. **OSC for Real-Time Communication**: OSC provides low-latency, structured data transfer between Python and TouchDesigner. Hierarchical address patterns (``/pose/{id}``, ``/hand/left/{id}``) enable easy filtering.

4. **VIDEO Mode Requires Timestamps**: When processing video, use ``detect_for_video()`` with consistent timestamps for temporal tracking.

Common Pitfalls
---------------

- **Missing models**: Run ``model_downloader.py`` first before any exercise
- **Wrong port**: Ensure TouchDesigner OSC In CHOP matches Python sender port (7000)
- **IMAGE vs VIDEO mode**: Use VIDEO mode for webcam (requires timestamps), IMAGE for single images
- **Coordinate system mismatch**: MediaPipe uses normalized 0-1; TD often uses pixel coordinates
- **BGR vs RGB**: OpenCV reads BGR; MediaPipe expects RGB (use ``cv2.cvtColor``)


Next Steps
==========

Continue to :doc:`../13.1.2_runwayml_bridge/README` to learn about integrating RunwayML models with TouchDesigner, or explore :doc:`../../13.2_realtime_ai_effects/13.2.3_pose_driven_effects/README` to apply your MediaPipe skills to advanced pose-driven visual effects.


References
==========

.. [Lugaresi2019] Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M., Zhang, F., Chang, C., Yong, M. G., Lee, J., Chang, W., Hua, W., Georg, M., & Grundmann, M. (2019). MediaPipe: A Framework for Building Perception Pipelines. *arXiv preprint arXiv:1906.08172*. https://arxiv.org/abs/1906.08172

.. [Bazarevsky2020] Bazarevsky, V., Grishchenko, I., Raveendran, K., Zhu, T., Zhang, F., & Grundmann, M. (2020). BlazePose: On-device Real-time Body Pose Tracking. *arXiv preprint arXiv:2006.10204*. https://arxiv.org/abs/2006.10204

.. [Zhang2020] Zhang, F., Bazarevsky, V., Vakunov, A., Tkachenka, A., Sung, G., Chang, C., & Grundmann, M. (2020). MediaPipe Hands: On-device Real-time Hand Tracking. *arXiv preprint arXiv:2006.10214*. https://arxiv.org/abs/2006.10214

.. [Tan2020] Tan, M., Pang, R., & Le, Q. V. (2020). EfficientDet: Scalable and Efficient Object Detection. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 10781-10790).

.. [Wright2005] Wright, M. (2005). Open Sound Control: An Enabling Encoding for Media Applications. In *Proceedings of the 2005 International Computer Music Conference*.

.. [Cao2019] Cao, Z., Hidalgo, G., Simon, T., Wei, S. E., & Sheikh, Y. (2019). OpenPose: Realtime Multi-Person 2D Pose Estimation Using Part Affinity Fields. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 43(1), 172-186.

.. [Derivative2024] Derivative. (2024). Python in TouchDesigner. *TouchDesigner Documentation*. https://docs.derivative.ca/Python

.. [Wigdor2011] Wigdor, D., & Wixon, D. (2011). *Brave NUI World: Designing Natural User Interfaces for Touch and Gesture*. Morgan Kaufmann.

.. [Edmonds2011] Edmonds, E. (2011). The Art of Interaction. *Digital Creativity*, 21(4), 257-264.

.. [MediaPipe2024] Google. (2024). MediaPipe Solutions. *MediaPipe Documentation*. https://developers.google.com/mediapipe/solutions
