5.3.1 - Bouncing Ball Animation
======================

Interactive real-time animation of a bouncing ball using NumPy and OpenCV.

Overview
--------

This demo creates a physics-based bouncing ball simulation that runs in real-time. The ball bounces around a window with collision detection and random velocity changes.

Features
--------

- Real-time rendering with OpenCV
- Physics simulation with collision detection
- NumPy-based distance calculations and masking
- Interactive controls (press 'q' to quit)
- Random velocity changes on collisions

Running the Demo
----------------

Make sure you have the required dependencies:

.. code-block:: bash

   pip install opencv-python numpy

Then run the script:

.. code-block:: bash

   python bouncing_ball.py

Controls
--------

- **q**: Quit the application

Technical Details
-----------------

The demo uses:

- **NumPy**: For mathematical operations and array manipulation
- **OpenCV**: For real-time window display and user input
- **Distance calculations**: Using ``np.ogrid`` for efficient pixel operations
- **Masking**: To create circular ball shapes

The ball's movement is calculated using simple physics with collision detection at window boundaries.
