================================================
10.2.2: Boids Flocking in TouchDesigner
================================================

:Duration: 35 minutes
:Level: Intermediate
:Prerequisites: Module 5.2.1 (Boids Flocking), Module 10.1 (TD Environment)

The Big Question
================

How do we transform a 300-line NumPy simulation into a real-time TouchDesigner system that handles thousands of agents at 60fps?

In :doc:`Module 5.2.1 </content/Module_05_simulation_emergent_behavior/5.2_flocking_swarms/5.2.1_boids/README>`, you implemented Craig Reynolds' boids algorithm using NumPy arrays and PIL rendering. That approach works well for generating offline animations, but what if you want to create an interactive installation where visitors can influence the flock in real-time? What if you need 5,000 boids instead of 50?

This exercise bridges your NumPy knowledge to TouchDesigner's real-time paradigm. You will learn how the same three rules (separation, alignment, cohesion) translate into TD's operator-based architecture, and why this translation enables dramatically better performance.

.. figure:: boids_td_demo.gif
   :width: 500px
   :align: center
   :alt: Animated simulation showing 50 boids flocking in real-time TouchDesigner style

   Boids simulation output demonstrating the same emergent behavior from Module 5.2.1, now structured for TouchDesigner's real-time architecture.

**Learning Objectives**

By completing this exercise, you will:

* Translate the NumPy boids algorithm to TouchDesigner's operator paradigm
* Understand how Script CHOPs handle physics computation in real-time
* Learn GPU instancing for efficient rendering of thousands of agents
* Compare performance characteristics between CPU-based NumPy and GPU-accelerated TD


Prerequisites Review
====================

This exercise builds directly on your work in Module 5.2.1. Before continuing, ensure you understand:

**The Three Boids Rules** (from Module 5.2.1)

1. **Separation**: Steer away from neighbors that are too close
2. **Alignment**: Match the average velocity of nearby neighbors
3. **Cohesion**: Steer toward the center of mass of nearby neighbors

.. figure:: ../../../Module_05_simulation_emergent_behavior/5.2_flocking_swarms/5.2.1_boids/boids_rules_diagram.png
   :width: 600px
   :align: center
   :alt: Three panels showing separation, alignment, and cohesion rules

   The three boids rules from Module 5.2.1. These same rules apply in TouchDesigner.

**TouchDesigner Basics** (from Module 10.1)

* Understanding of the four operator families: TOP, CHOP, SOP, DAT
* Basic navigation of the TouchDesigner interface
* Familiarity with the cook cycle and real-time execution

If you need a refresher on boids fundamentals, review :doc:`Module 5.2.1 </content/Module_05_simulation_emergent_behavior/5.2_flocking_swarms/5.2.1_boids/README>` before continuing.


Part 1: NumPy to TouchDesigner Translation
==========================================

The fundamental challenge when porting from NumPy to TouchDesigner is understanding how each system organizes data and computation. Let us examine the architectural differences.

NumPy vs TouchDesigner Architecture
-----------------------------------

In NumPy, you work with arrays that you manipulate sequentially in a Python script. In TouchDesigner, data flows through a network of connected operators, with each operator performing a specific transformation.

.. figure:: numpy_vs_td_comparison.png
   :width: 700px
   :align: center
   :alt: Side-by-side comparison of NumPy and TouchDesigner architectures

   Architectural comparison showing how the boids pipeline maps from NumPy to TouchDesigner operators.

**Key Translation Mappings**

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - NumPy Concept
     - TouchDesigner Equivalent
     - Notes
   * - NumPy array ``positions[50, 2]``
     - Table DAT with 50 rows
     - Columns: pos_x, pos_y
   * - Python for-loop
     - Script CHOP ``onCook()``
     - Executes every frame
   * - ``np.sqrt()``, ``np.sum()``
     - NumPy in TD Python
     - Same syntax inside TD
   * - PIL ``ImageDraw``
     - Geometry COMP + Render TOP
     - GPU-accelerated
   * - ``imageio.mimsave()``
     - Movie File Out TOP
     - Or direct output to display

Data Storage: Arrays vs Tables
------------------------------

In NumPy, boid state is stored in arrays:

.. code-block:: python
   :caption: NumPy data storage (from Module 5.2.1)

   # Positions: shape (50, 2) - each row is [x, y]
   positions = np.random.rand(50, 2) * 512

   # Velocities: shape (50, 2) - each row is [vx, vy]
   velocities = np.column_stack([np.cos(angles), np.sin(angles)])

In TouchDesigner, you would store this in a Table DAT or use persistent storage in a Script CHOP:

.. code-block:: python
   :caption: TouchDesigner data storage (Script CHOP)

   # Initialize state in Script CHOP storage
   if 'positions' not in scriptOp.storage:
       scriptOp.storage['pos_x'] = np.random.rand(50) * 512
       scriptOp.storage['pos_y'] = np.random.rand(50) * 512
       scriptOp.storage['vel_x'] = np.cos(angles)
       scriptOp.storage['vel_y'] = np.sin(angles)

The key difference: TouchDesigner storage persists across frames automatically, while NumPy arrays only exist within your script's execution.

.. note:: SCREENSHOT NEEDED: TD network showing basic boids setup

   **Screenshot instructions**: Create a TouchDesigner screenshot showing:
   - A Table DAT named ``boids_state`` with columns: id, pos_x, pos_y, vel_x, vel_y
   - A Script CHOP connected to the table
   - The network layout showing data flow from DAT to CHOP


Part 2: Script-Based Physics in TouchDesigner
=============================================

The physics computation in TouchDesigner happens inside a Script CHOP or Execute DAT. The ``onCook()`` callback runs every frame, making it ideal for simulation updates.

The Physics Update Loop
-----------------------

Here is how the boids physics translates to TouchDesigner:

.. code-block:: python
   :caption: Script CHOP physics computation
   :linenos:

   def onCook(scriptOp):
       """Main cook callback - runs every frame at 60fps."""
       import numpy as np

       # Read current state from storage
       pos_x = scriptOp.storage['pos_x']
       pos_y = scriptOp.storage['pos_y']
       positions = np.column_stack([pos_x, pos_y])

       # Compute distances (same as Module 5.2.1)
       differences = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
       distances = np.sqrt(np.sum(differences ** 2, axis=2))

       # Calculate forces (separation, alignment, cohesion)
       # ... same logic as Module 5.2.1 ...

       # Update positions and velocities
       positions = positions + velocities
       positions = positions % 512  # Wrap at boundaries

       # Write back to storage
       scriptOp.storage['pos_x'] = positions[:, 0]
       scriptOp.storage['pos_y'] = positions[:, 1]

       # Output channels for rendering
       scriptOp.clear()
       pos_x_chan = scriptOp.appendChan('pos_x')
       for i in range(len(positions)):
           pos_x_chan[i] = positions[i, 0]

The structure mirrors your NumPy code from Module 5.2.1, but wrapped in TD's callback system.

**Key Parameters** (same as Module 5.2.1)

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Value
     - Effect
   * - ``PERCEPTION_RADIUS``
     - 50
     - How far each boid can see neighbors
   * - ``SEPARATION_WEIGHT``
     - 1.5
     - Strength of collision avoidance
   * - ``ALIGNMENT_WEIGHT``
     - 1.0
     - Strength of velocity matching
   * - ``COHESION_WEIGHT``
     - 1.0
     - Strength of flock cohesion
   * - ``MAX_SPEED``
     - 4
     - Maximum velocity magnitude

.. note:: SCREENSHOT NEEDED: Script CHOP with Python code visible

   **Screenshot instructions**: Create a TouchDesigner screenshot showing:
   - The Script CHOP's DAT panel open with Python code visible
   - The ``onCook`` function visible
   - The Performance Monitor showing cook time (Alt+Y)

Performance Considerations
--------------------------

TouchDesigner's cook cycle introduces specific performance constraints:

1. **Cook Time Budget**: At 60fps, each frame has ~16.6ms total. Your Script CHOP must complete within this budget.

2. **Python GIL**: Python operations in TD are single-threaded. For complex physics, consider GLSL compute shaders.

3. **Storage vs. Input**: Using ``scriptOp.storage`` is faster than reading from connected DATs every frame.

.. important::

   Start with 50-100 boids in Script CHOP Python. For thousands of boids, you will need GLSL-based computation (covered in Module 10.3).


Part 3: GPU Instancing for Rendering
====================================

The biggest performance gain when moving to TouchDesigner comes from GPU instancing. Instead of drawing each boid individually (like PIL's ``draw.polygon()``), you render one piece of geometry and let the GPU replicate it at each boid position.

Why Instancing?
---------------

In NumPy/PIL, rendering 50 boids means 50 separate draw calls:

.. code-block:: python
   :caption: PIL rendering (Module 5.2.1) - CPU bound

   for i in range(len(positions)):
       x, y = positions[i]
       draw.polygon([...], fill=color)  # One draw call per boid

In TouchDesigner, you render once and instance at all positions:

.. code-block:: text

   1. Geometry COMP contains a single Cone SOP
   2. Instance CHOP provides 50 position/rotation values
   3. GPU draws all 50 cones in a single draw call

This is why TouchDesigner can render 5,000+ boids at 60fps while NumPy/PIL struggles with 100.

.. figure:: td_network_diagram.png
   :width: 600px
   :align: center
   :alt: TouchDesigner node network showing instancing setup

   Simplified TouchDesigner network showing how Script CHOP output feeds into GPU instancing for rendering.

Setting Up Instancing
---------------------

The instancing pipeline in TouchDesigner:

1. **Script CHOP outputs channels**: ``pos_x``, ``pos_y``, ``pos_z``, ``rot_z``
2. **Geometry COMP enables instancing**: Check "Instancing" on the Instance page
3. **Map channels to transforms**: tx=pos_x, ty=pos_y, rz=rot_z
4. **Render TOP captures the output**: Camera + Light + Render

.. note:: SCREENSHOT NEEDED: Instance COMP settings

   **Screenshot instructions**: Create a TouchDesigner screenshot showing:
   - Geometry COMP's Instance page with Instancing enabled
   - Channel mapping: tx, ty, tz to pos_x, pos_y, pos_z
   - rz mapped to rotation channel
   - The Instance CHOP reference


Synthesis Project: Add Obstacle Avoidance
=========================================

Now apply your understanding by implementing the fourth rule from Module 5.2.1: obstacle avoidance.

**Goal**

Implement an ``obstacle_avoidance()`` function that makes boids steer away from a circular obstacle in the center of the canvas. This mirrors Exercise 3 from Module 5.2.1, adapted for the TouchDesigner architecture.

**Requirements**

1. Boids within 1.5x the obstacle radius should steer away
2. The avoidance force should be stronger when closer to the obstacle
3. The force should point directly away from the obstacle center

**Starter Code**

Download and run the starter script:

:download:`Download boids_td_starter.py <boids_td_starter.py>`

Open ``boids_td_starter.py`` and find the ``obstacle_avoidance()`` function with TODO markers.

.. dropdown:: Hint 1: Distance to Obstacle

   Calculate the distance from each boid to the obstacle center:

   .. code-block:: python

      obstacle_center = np.array([OBSTACLE_X, OBSTACLE_Y])
      direction = positions[i] - obstacle_center
      distance = np.sqrt(np.sum(direction ** 2))

.. dropdown:: Hint 2: Avoidance Force

   The force should point away and be stronger when closer:

   .. code-block:: python

      if distance < OBSTACLE_RADIUS * 1.5 and distance > 0:
          normalized = direction / distance
          strength = (OBSTACLE_RADIUS * 1.5 - distance) / distance
          steering[i] = normalized * strength

.. dropdown:: Complete Solution

   .. code-block:: python

      def obstacle_avoidance(positions):
          """Steer away from the central obstacle."""
          steering = np.zeros_like(positions)
          obstacle_center = np.array([OBSTACLE_X, OBSTACLE_Y])

          for i in range(len(positions)):
              direction = positions[i] - obstacle_center
              distance = np.sqrt(np.sum(direction ** 2))

              # Avoid if within 1.5x obstacle radius
              if distance < OBSTACLE_RADIUS * 1.5 and distance > 0:
                  # Force pointing away, stronger when closer
                  normalized = direction / distance
                  strength = (OBSTACLE_RADIUS * 1.5 - distance) / distance
                  steering[i] = normalized * strength

          return steering

.. note:: SCREENSHOT NEEDED: Obstacle avoidance in TD

   **Screenshot instructions**: After implementing in TouchDesigner:
   - Show the network with obstacle avoidance added
   - Include a Circle SOP for the obstacle visualization
   - Capture boids steering around the obstacle

**TouchDesigner Implementation**

To implement this in actual TouchDesigner:

1. Add the obstacle avoidance function to your Script CHOP
2. Create a Circle SOP for visualizing the obstacle
3. Merge the obstacle geometry with your instanced boids
4. Optionally add a mouse-in CHOP to make the obstacle follow the cursor


Summary
=======

**Key Takeaways**

* TouchDesigner uses an **operator-based paradigm** where data flows through connected nodes, unlike NumPy's sequential script execution
* **Script CHOPs** provide a bridge for NumPy-style computation within TD's real-time cook cycle
* **GPU instancing** enables rendering thousands of agents efficiently by drawing geometry once and replicating it at multiple positions
* The **same boids algorithm** from Module 5.2.1 applies directly in TD, just wrapped in different architectural patterns
* **Performance scales differently**: NumPy is CPU-bound and sequential; TD leverages GPU parallelism

**NumPy vs TouchDesigner Performance**

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Metric
     - NumPy/PIL
     - TouchDesigner
   * - Typical boid count
     - 50-200
     - 1,000-10,000+
   * - Frame rate
     - Variable (offline)
     - Locked 60fps
   * - Rendering
     - CPU (PIL draw calls)
     - GPU (instancing)
   * - Interactivity
     - None (pre-rendered)
     - Real-time control
   * - Best for
     - Prototyping, offline art
     - Installations, live performance

**Common Pitfalls**

* **Forgetting storage initialization**: Always check ``if 'key' not in scriptOp.storage`` before first use
* **Exceeding cook time budget**: Profile with Performance Monitor (Alt+Y) if simulation stutters
* **Not clearing Script CHOP output**: Call ``scriptOp.clear()`` before appending new channels
* **Incorrect instance channel mapping**: Double-check tx/ty/tz and rotation mappings on Geometry COMP


What's Next
===========

Continue to :doc:`Module 10.2.3 </content/Module_10_touchdesigner_fundamentals/10.2_recreating_static_exercises/10.2.3_planet_simulation_td/README>` to port the N-body planet simulation to TouchDesigner, learning about feedback loops for orbital trails.

For deeper exploration:

* **Module 10.3**: NumPy-TD Pipeline covers Script Operators and custom components
* **Module 11.1**: Input Devices shows how to make boids react to webcam or audio
* **GLSL optimization**: For 10,000+ boids, explore GLSL TOPs for parallel computation


References
==========

.. [Reynolds1987TD] Reynolds, C. W. (1987). Flocks, herds and schools: A distributed behavioral model. *ACM SIGGRAPH Computer Graphics*, 21(4), 25-34. https://doi.org/10.1145/37402.37406 [Original boids paper]

.. [Reynolds1999TD] Reynolds, C. W. (1999). Steering behaviors for autonomous characters. *Proceedings of Game Developers Conference 1999*, 763-782. https://www.red3d.com/cwr/steer/ [Practical steering implementation]

.. [Derivative2024] Derivative. (2024). TouchDesigner Documentation. https://derivative.ca/UserGuide/ [Official TD documentation]

.. [DerivativeWiki] Derivative. (2024). Script CHOP. *TouchDesigner Wiki*. https://docs.derivative.ca/Script_CHOP [Script CHOP reference]

.. [GPUGems2004] Pharr, M., & Fernando, R. (Eds.). (2004). GPU Gems: Programming techniques, tips, and tricks for real-time graphics. Addison-Wesley. [GPU instancing techniques]

.. [ShiffmanTD2012] Shiffman, D. (2012). *The Nature of Code*. Self-published. https://natureofcode.com/ [Excellent pedagogical resource for simulations]

.. [Vicsek2012TD] Vicsek, T., & Zafeiris, A. (2012). Collective motion. *Physics Reports*, 517(3-4), 71-140. https://doi.org/10.1016/j.physrep.2012.03.004 [Physics of flocking behavior]

.. [TDCommunity] TouchDesigner Community. (2024). Instancing tutorials and examples. *Derivative Forums*. https://forum.derivative.ca/ [Community resources and patterns]
