=====================================
10.1.2 - Python Integration Basics
=====================================

:Duration: 35-40 minutes
:Level: Intermediate (requires Module 10.1.1)
:Software: TouchDesigner 2025.31310, Python 3.11

Overview
========

Real-time creative tools need more than static scripts. You have spent Modules 1-9 mastering NumPy arrays, writing Python scripts that process images, and generating visuals through code. But what happens when you need your Python logic to control a live, continuously running visual network?

This is where **Python integration in TouchDesigner** becomes essential. Unlike standalone Python that runs once and stops, TouchDesigner embeds a full Python interpreter that can interact with every node in your network, executing code 60+ times per second.

In this exercise, you will learn where Python code lives in TouchDesigner, how to reference operators with the op() function, and how to read or modify parameters programmatically. Your Python knowledge from Modules 0-9 transfers directly. You just need to learn the TouchDesigner-specific syntax.

**Prerequisites**

Before beginning, ensure you have:

* Completed Module 10.1.1 (Understanding of TOPs, CHOPs, SOPs, DATs, and the cooking concept)
* TouchDesigner 2025.31310 installed (free non-commercial version from `derivative.ca <https://derivative.ca/download>`_)
* Python 3.11 installed (bundled with TouchDesigner)
* Basic Python proficiency from Modules 0-9

**Learning Objectives**

By completing this exercise, you will:

* Identify where Python code executes in TouchDesigner (Script DAT, Script CHOP, Script TOP)
* Use the op() function to reference any operator in your network
* Read and modify operator parameters using the .par syntax
* Build interactive systems where Python controls visual output in real-time

Quick Start: See Python Integration in Action
==============================================

Before diving into syntax details, let's see Python controlling TouchDesigner visually. Below is a Script DAT creating an animated pattern by controlling multiple parameters simultaneously:

.. figure:: quick_start_animation.gif
   :width: 60%
   :align: center
   :alt: Animated kaleidoscope pattern with noise, blur, and brightness rhythmically changing based on Python-controlled sine waves

   A Python script controlling five parameters in real-time. The script uses sine waves and time to create rhythmic visual patterns. This is what "Python driving nodes" looks like. [PLACEHOLDER - USER WILL CREATE]

**What's happening here?**

This network uses a Script DAT that runs every frame. The Python code reads the current time, calculates oscillating values using math.sin(), and applies those values to control:

* Noise TOP period (size of noise pattern)
* Noise TOP amplitude (intensity)
* Blur TOP size (amount of blur)
* Level TOP brightness (overall luminosity)

Unlike standalone Python that runs once, this script executes 60+ times per second, creating smooth animation. Change any value in the script, and the pattern instantly responds. This is the power of Python integration.

You will build this exact network in Exercise 2. First, let's understand the fundamental concepts that make it possible.

Core Concept 1: Where Python Lives in TouchDesigner
====================================================

Script Operators Overview
^^^^^^^^^^^^^^^^^^^^^^^^^

Unlike standalone Python where you write code in a file and run it once, TouchDesigner embeds Python directly into the node network. Python code lives inside special **Script operators**:

.. figure:: script_operator_types.png
   :width: 100%
   :align: center
   :alt: Three types of script operators: Script DAT for text and control, Script CHOP for numeric channels, and Script TOP for pixel data

   TouchDesigner provides three main script operator types. Script DAT is the most versatile and commonly used for general Python programming. Diagram generated with Claude - Opus 4.5.

**Script DAT** is your primary tool for Python programming in TouchDesigner. It executes Python code and can output text or table data. Use it for:

- Control logic and automation
- Reading values from operators
- Modifying parameters across your network
- Processing external data

**Script CHOP** generates numeric channel data using Python. The code runs every frame and must return numeric values. Use it for:

- Custom animation curves
- Mathematical signal generation
- Audio processing algorithms

**Script TOP** generates or processes pixel data. It can create images from NumPy arrays or manipulate incoming image data. Use it for:

- Custom image generation
- Complex pixel operations
- Integrating NumPy image processing code

.. tip::

   Start with Script DAT for general Python programming. Only use Script CHOP or Script TOP when you specifically need to output channels or images.


The Textport: Your Python Console
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TouchDesigner includes a built-in Python console called the **Textport**. Access it through:

- **Dialogs > Textport** from the menu
- **Alt+T** keyboard shortcut

The Textport shows output from ``print()`` statements and displays error messages. When your Script DAT runs, any printed text appears here.

Try opening the Textport now and typing:

::

   print("Hello from TouchDesigner!")

You should see the message appear immediately. The Textport is essential for debugging your Python code.

.. figure:: textport_screenshot.png
   :width: 70%
   :align: center
   :alt: TouchDesigner Textport window showing Python print() output and error messages with colored syntax highlighting

   The Textport (Alt+T) displays all Python output. Use it for debugging and monitoring your scripts. [PLACEHOLDER - USER WILL CREATE SCREENSHOT]


Execution Model: Frame-Based vs Run-Once
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here is the critical difference between standalone Python and TouchDesigner Python:

.. figure:: execution_model.png
   :width: 100%
   :align: center
   :alt: Comparison diagram showing standalone Python running once top-to-bottom versus TouchDesigner Python running in a continuous 60fps loop

   Standalone Python executes once and stops. TouchDesigner Python runs continuously, executing your code every frame (typically 60 times per second). Diagram generated with Claude - Opus 4.5.

In a Script DAT, your code can run:

1. **Once** - When you click "Run" or when the operator first cooks
2. **Every Frame** - If you enable "Run on Start" and the DAT has inputs that update

.. list-table:: Python Execution Comparison
   :header-rows: 1
   :widths: 40 30 30

   * - Aspect
     - Standalone Python
     - TouchDesigner Python
   * - Execution trigger
     - You run the script
     - Frame update or event
   * - Variable persistence
     - Lost when script ends
     - Can persist in ``mod.name``
   * - Output destination
     - Console/files
     - Textport/operators
   * - Import availability
     - All installed packages
     - TD-bundled + installed


.. important::

   When writing Python in TouchDesigner, remember that your code may run 60+ times per second. Avoid expensive operations like file I/O in frame-based scripts.

.. figure:: script_dat_parameters.png
   :width: 80%
   :align: center
   :alt: Script DAT parameter panel showing Run button, Run on Start toggle, and callback configuration options

   Script DAT parameters control when your Python code executes. Enable "Run on Start" for frame-based execution. [PLACEHOLDER - USER WILL CREATE SCREENSHOT]


Core Concept 2: The op() Function - Your Gateway to Nodes
==========================================================

Basic op() Syntax
^^^^^^^^^^^^^^^^^

The ``op()`` function is how you reference any operator from Python. It returns an operator object that you can query or modify:

.. code-block:: python
   :caption: Basic operator reference
   :linenos:

   # Reference an operator by name
   noise_operator = op('noise1')

   # Print information about it
   print(noise_operator.name)     # Output: noise1
   print(noise_operator.type)     # Output: noisetoп
   print(noise_operator.family)   # Output: TOP

The ``op()`` function searches for operators by name within the current context. If you have a Noise TOP called "noise1" in your network, ``op('noise1')`` returns a reference to it.


The ``me`` Object: Self-Reference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Inside any operator's Python code, ``me`` refers to the operator itself:

.. code-block:: python
   :caption: Using me for self-reference
   :linenos:

   # Inside a Script DAT called "script1"
   print(me.name)        # Output: script1
   print(me.type)        # Output: textDAT
   print(me.path)        # Output: /project1/script1

   # Access the parent container
   parent_op = me.parent()
   print(parent_op.name)  # Output: project1

The ``me`` object is essential when your script needs to know its own identity or navigate relative to its position in the network.


Operator Properties
^^^^^^^^^^^^^^^^^^^

Every operator has properties you can read:

.. code-block:: python
   :caption: Accessing operator properties
   :linenos:

   # Get a reference to a TOP
   blur = op('blur1')

   # Read image dimensions (TOP-specific)
   print(blur.width)      # Image width in pixels
   print(blur.height)     # Image height in pixels

   # Check if the operator is cooking
   print(blur.isCooking)  # True/False

   # Get the operator's path
   print(blur.path)       # Full path like /project1/blur1

Different operator families have different properties. TOPs have ``.width`` and ``.height``, CHOPs have ``.numChans`` and ``.numSamples``, and so on. The TouchDesigner documentation provides a complete reference (Derivative Inc., 2024).


Path References
^^^^^^^^^^^^^^^

When operators are in different containers, you need to use paths:

.. figure:: op_reference_paths.png
   :width: 100%
   :align: center
   :alt: Diagram showing operator hierarchy with examples of absolute and relative path references

   Operators are organized in a hierarchy. Use relative paths (../) or absolute paths (/project1/container1/op) to reference operators in different locations. Diagram generated with Claude - Opus 4.5.

.. code-block:: python
   :caption: Different path reference methods
   :linenos:

   # Absolute path - always works, starts with /
   noise = op('/project1/container1/noise1')

   # Relative path - from current location
   sibling = op('blur1')           # Same container
   parent_container = op('..')      # Go up one level
   cousin = op('../container2/op')  # Sibling container's child

   # Using parent() to navigate
   parent = me.parent()
   grandparent = me.parent().parent()


The Python Connection
^^^^^^^^^^^^^^^^^^^^^

Your Python knowledge from Modules 0-9 transfers directly to TouchDesigner, with one key addition: the op() function.

.. list-table:: Python to TouchDesigner op() Mapping
   :header-rows: 1
   :widths: 50 50

   * - Standalone Python
     - TouchDesigner Python
   * - ``my_object = MyClass()``
     - ``my_operator = op('operator_name')``
   * - ``my_object.method()``
     - ``my_operator.cook()``
   * - ``my_object.property``
     - ``my_operator.width``
   * - ``import module``
     - ``import module`` (same!)
   * - ``print(value)``
     - ``print(value)`` (outputs to Textport)

Everything else is standard Python. The only new syntax is op() for referencing nodes.

.. figure:: network_hierarchy_screenshot.png
   :width: 90%
   :align: center
   :alt: TouchDesigner network pane showing nested containers with multiple operators and visible path structure

   The network hierarchy in TouchDesigner. Containers organize operators like folders organize files. [PLACEHOLDER - USER WILL CREATE SCREENSHOT]


.. admonition:: Did You Know?

   TouchDesigner's ``op()`` function was inspired by similar referencing systems in other visual programming environments, following established patterns in dataflow programming (Hils, 1992). The ability to access any node from any script creates a powerful bridge between visual and textual programming paradigms.


Core Concept 3: Reading and Writing Parameters
==============================================

Parameter Access with .par
^^^^^^^^^^^^^^^^^^^^^^^^^^

Every parameter on an operator can be accessed through the ``.par`` object:

.. code-block:: python
   :caption: Accessing parameters
   :linenos:

   # Get a reference to Blur TOP
   blur = op('blur1')

   # Read the current size parameter value
   current_size = blur.par.size.eval()
   print(f"Current blur size: {current_size}")

   # Read the filtertype parameter
   filter_type = blur.par.filtertype.eval()
   print(f"Filter type: {filter_type}")

Parameter names match what you see in the parameter panel, using lowercase letters. You can hover over any parameter in TouchDesigner to see its internal name.


Reading vs Writing Values
^^^^^^^^^^^^^^^^^^^^^^^^^

To read a parameter's current value, use ``.eval()``:

.. code-block:: python
   :caption: Reading parameter values
   :linenos:

   # .eval() returns the current computed value
   blur = op('blur1')
   size = blur.par.size.eval()      # Returns a number
   name = blur.par.name.eval()      # Returns a string

To write a parameter, assign directly to it:

.. code-block:: python
   :caption: Writing parameter values
   :linenos:

   # Direct assignment sets the parameter
   blur = op('blur1')
   blur.par.size = 10              # Set blur size to 10
   blur.par.filtertype = 'gauss'   # Set filter type

You can also set parameters to expressions:

.. code-block:: python
   :caption: Setting parameter expressions
   :linenos:

   # Set a parameter to a Python expression
   blur = op('blur1')
   blur.par.size.expr = "me.time.seconds * 5"  # Animates over time

.. warning::

   When setting parameters, the change happens immediately and propagates through the network. Be careful not to create infinite loops where a script modifies parameters that cause the script to run again.


Common Parameter Examples
^^^^^^^^^^^^^^^^^^^^^^^^^

Here are practical examples with commonly used operators:

.. code-block:: python
   :caption: Working with Noise TOP parameters
   :linenos:

   noise = op('noise1')

   # Adjust noise characteristics
   noise.par.type = 'random'     # Type: random, sparse, harmonic, etc.
   noise.par.period = 100        # Size of noise pattern
   noise.par.amp = 1.5           # Amplitude multiplier

   # Animate the offset
   noise.par.tx = me.time.seconds * 10  # Pan noise over time

.. code-block:: python
   :caption: Working with Level TOP parameters
   :linenos:

   level = op('level1')

   # Adjust brightness and contrast
   level.par.brightness = 0.2    # -1 to 1 range
   level.par.contrast = 1.5      # 0 to 2+ range
   level.par.opacity = 0.8       # 0 to 1 range

.. code-block:: python
   :caption: Conditional parameter control
   :linenos:

   # Read a value from a CHOP and use it to control a TOP
   lfo_value = op('lfo1')[0].eval()  # Get first channel value

   blur = op('blur1')
   if lfo_value > 0:
       blur.par.size = lfo_value * 20
   else:
       blur.par.size = 0


Visual Parameter Control
^^^^^^^^^^^^^^^^^^^^^^^^

Here's a complete example showing how Python creates animated effects:

.. code-block:: python
   :caption: Animating parameters with Python
   :linenos:
   :emphasize-lines: 6-8, 13-15

   import math

   # Get current time
   t = me.time.seconds

   # Create pulsing blur effect
   pulse = abs(math.sin(t * 2))  # Oscillates 0 to 1
   blur_size = pulse * 20

   # Apply to Blur TOP
   blur = op('blur1')
   blur.par.size = blur_size

   # Also control Noise period inversely
   noise = op('noise1')
   noise.par.period = 100 - (pulse * 50)  # Smaller when blur is larger

This creates a coordinated animation where blur and noise period move in opposite directions, creating a breathing effect.

.. figure:: parameter_panel_screenshot.png
   :width: 70%
   :align: center
   :alt: TouchDesigner parameter panel with mouse cursor hovering over a parameter, showing tooltip with internal name

   Hover over any parameter to see its internal name. This is what you use in Python code. [PLACEHOLDER - USER WILL CREATE SCREENSHOT]

.. figure:: parameter_animation.gif
   :width: 50%
   :align: center
   :alt: Animated visualization showing blur size parameter slider moving automatically as Python script controls it in real-time

   Parameters being controlled live by Python. Notice how changes happen smoothly every frame. [PLACEHOLDER - USER WILL CREATE GIF]


Hands-On Exercises
------------------

Now let us put these concepts into practice. Open TouchDesigner and follow along with these exercises.


Exercise 1: Explore the Textport (5 min)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Goal**: Create a Script DAT that prints operator information.

**Instructions**:

1. Create a new TouchDesigner project
2. Press **Tab** > **DAT** > **Text** to create a Text DAT
3. Press **Tab** > **TOP** > **Noise** to create a Noise TOP
4. Press **Tab** > **TOP** > **Blur** and connect it to the Noise TOP
5. Select the Text DAT and rename it to "script1" in the parameters
6. In the Text DAT, type the following code:

.. code-block:: python
   :linenos:

   # Explore operators in our network
   noise = op('noise1')
   blur = op('blur1')

   print("=== Noise TOP Info ===")
   print(f"Name: {noise.name}")
   print(f"Type: {noise.type}")
   print(f"Dimensions: {noise.width} x {noise.height}")

   print("\n=== Blur TOP Info ===")
   print(f"Name: {blur.name}")
   print(f"Current size: {blur.par.size.eval()}")

7. Open the Textport (**Alt+T**)
8. Right-click on the Text DAT and select **Run Script**

.. dropdown:: Reflection Questions

   1. What information did the script print about each operator?
   2. What happens if you change the Blur size slider and run the script again?
   3. How would you modify the script to also print the Noise TOP's period parameter?


Exercise 2: Parameter Controller (7 min)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Goal**: Modify a script that controls parameters based on time.

**Starting Network**:

- Noise TOP > Blur TOP > Null TOP (for viewing)
- Text DAT for your script

**Instructions**:

1. Use your network from Exercise 1
2. Edit your Text DAT with this code:

.. code-block:: python
   :linenos:

   import math

   # Get current time in seconds
   current_time = me.time.seconds

   # Calculate a sine wave value (oscillates between -1 and 1)
   sine_value = math.sin(current_time)

   # Map the sine value to blur size (0 to 20)
   blur_size = (sine_value + 1) * 10  # Now ranges from 0 to 20

   # Apply to the blur operator
   blur = op('blur1')
   blur.par.size = blur_size

   print(f"Time: {current_time:.2f}, Blur: {blur_size:.2f}")

3. Right-click the Text DAT and enable **Run on Start**
4. Press **F1** to reset the timeline
5. Observe the blur changing over time

**Your Task**: Modify the script to:

- Change the oscillation speed (hint: multiply ``current_time``)
- Add control for Noise TOP's period parameter

.. dropdown:: Hint

   To change oscillation speed, multiply the time value:

   .. code-block:: python

      sine_value = math.sin(current_time * 2)  # Twice as fast

   To control the Noise period:

   .. code-block:: python

      noise = op('noise1')
      noise.par.period = 50 + (sine_value + 1) * 50  # Range 50-150

.. dropdown:: Solution

   .. code-block:: python
      :linenos:

      import math

      current_time = me.time.seconds

      # Faster oscillation
      sine_value = math.sin(current_time * 2)

      # Control blur size
      blur = op('blur1')
      blur.par.size = (sine_value + 1) * 10

      # Control noise period
      noise = op('noise1')
      noise.par.period = 50 + (sine_value + 1) * 50

      print(f"Blur: {blur.par.size.eval():.1f}, Noise Period: {noise.par.period.eval():.1f}")

.. figure:: exercise_2_output.gif
   :width: 50%
   :align: center
   :alt: Animated noise pattern with blur pulsing from subtle to strong, creating rhythmic breathing effect over several seconds

   Your Exercise 2 result: blur size smoothly oscillating based on time. The visual feedback helps you understand frame-based Python execution. [PLACEHOLDER - USER WILL CREATE]

**Expected Behavior**:

When you enable "Run on Start", you should see:

* Blur size smoothly pulsing between 0 and 20
* Textport printing changing values every frame
* Instant response when you modify oscillation speed


Exercise 3: Interactive System (8 min)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Goal**: Build a system where a CHOP controls parameters via Python.

**Network to Build**:

1. LFO CHOP (generates oscillating values)
2. Math CHOP (remaps the range)
3. Text DAT (reads CHOP, controls TOPs)
4. Noise TOP > Blur TOP > Level TOP > Null TOP

**Instructions**:

1. Create an **LFO CHOP** and set:

   - Type: Sin
   - Frequency: 0.5

2. Create a **Math CHOP** connected to the LFO:

   - Range: From -1 to 1, To 0 to 1

3. Create your TOP chain: Noise > Blur > Level > Null

4. Create a **Text DAT** with this starter code:

.. code-block:: python
   :linenos:

   # Read value from Math CHOP
   chop_value = op('math1')[0].eval()

   # TODO: Use chop_value to control Blur size (0-20 range)
   blur = op('blur1')
   # Your code here

   # TODO: Use chop_value to control Level brightness (-0.5 to 0.5 range)
   level = op('level1')
   # Your code here

   print(f"CHOP value: {chop_value:.3f}")

5. Complete the TODOs to make the CHOP control both operators

**Requirements**:

- Blur size should range from 0 to 20
- Level brightness should range from -0.5 to 0.5
- Enable "Run on Start" to see continuous animation

.. dropdown:: Solution

   .. code-block:: python
      :linenos:

      # Read value from Math CHOP (0 to 1 range)
      chop_value = op('math1')[0].eval()

      # Control Blur size (0-20 range)
      blur = op('blur1')
      blur.par.size = chop_value * 20

      # Control Level brightness (-0.5 to 0.5 range)
      level = op('level1')
      level.par.brightness = chop_value - 0.5

      print(f"CHOP: {chop_value:.3f}, Blur: {blur.par.size.eval():.1f}, Brightness: {level.par.brightness.eval():.2f}")

.. dropdown:: Challenge Extension

   Add conditional logic: when the CHOP value is above 0.7, switch the Noise type from "random" to "sparse":

   .. code-block:: python

      noise = op('noise1')
      if chop_value > 0.7:
          noise.par.type = 'sparse'
      else:
          noise.par.type = 'random'

.. figure:: exercise_3_output.gif
   :width: 60%
   :align: center
   :alt: Noise pattern with synchronized blur and brightness changes driven by LFO CHOP, creating smooth organic pulsing motion

   Your Exercise 3 result: CHOP values driving multiple TOP parameters. The smooth oscillation creates organic, breathing visuals. [PLACEHOLDER - USER WILL CREATE]

.. figure:: exercise_3_network.png
   :width: 90%
   :align: center
   :alt: Complete TouchDesigner network showing LFO CHOP connected to Math CHOP, Script DAT reading CHOP values, and TOP chain being controlled

   The complete Exercise 3 network. Notice how CHOP data flows into the Script DAT, which then controls TOP parameters. [PLACEHOLDER - USER WILL CREATE SCREENSHOT]

**Expected Behavior**:

When complete, you should see:

* Visual output smoothly animating based on LFO wave
* Blur amount and brightness changing in sync
* When LFO value is high, blur is strong and image is bright
* When LFO value is low, blur is subtle and image is darker


Summary
-------

Key Takeaways
^^^^^^^^^^^^^

1. **Python lives in Script operators** - Script DAT is your primary tool, Script CHOP generates channels, Script TOP processes pixels

2. **The Textport is your debugger** - Use ``print()`` statements and Alt+T to see output and errors

3. **The ``op()`` function references any operator** - Use ``op('name')`` for same container, paths for other locations

4. **The ``me`` object is self-reference** - Access the current operator's properties and navigate with ``me.parent()``

5. **Parameters use the ``.par`` syntax** - Read with ``.par.name.eval()``, write with ``.par.name = value``


Common Pitfalls
^^^^^^^^^^^^^^^

.. warning::

   - **Forgetting ``.eval()`` when reading**: ``blur.par.size`` returns a Parameter object, not the value. Use ``blur.par.size.eval()`` to get the actual number.

   - **Case sensitivity**: Parameter names are case-sensitive. Use the exact name from TouchDesigner (hover over parameter to see it).

   - **Running expensive code every frame**: If your script runs 60 times per second, avoid file I/O, network calls, or heavy computations.

   - **Circular references**: Be careful not to create scripts that modify parameters which trigger the same script to run again.


References
----------

.. [Derivative2024] Derivative Inc. (2024). *TouchDesigner Python Reference*. Retrieved from https://derivative.ca/UserGuide/Python [Official documentation for all Python integration in TouchDesigner]

.. [Derivative2024b] Derivative Inc. (2024). *Script DAT*. In TouchDesigner Documentation. Retrieved from https://derivative.ca/UserGuide/Script_DAT [Reference for Script DAT operator]

.. [Hils1992] Hils, D. D. (1992). Visual languages and computing survey: Data flow visual programming languages. *Journal of Visual Languages and Computing*, 3(1), 69-101. [Survey of dataflow programming paradigms and their scripting interfaces]

.. [ReasFry2014] Reas, C., & Fry, B. (2014). *Processing: A Programming Handbook for Visual Designers and Artists* (2nd ed.). MIT Press. [Foundational text on integrating code with visual creative tools]

.. [Burnett1994] Burnett, M. M., & Baker, M. J. (1994). A classification system for visual programming languages. *Journal of Visual Languages and Computing*, 5(3), 287-300. [Academic framework for understanding the relationship between visual and textual programming]

.. [GreenPetre1996] Green, T. R. G., & Petre, M. (1996). Usability analysis of visual programming environments: A 'cognitive dimensions' framework. *Journal of Visual Languages and Computing*, 7(2), 131-174. [Analysis of how users navigate between visual and scripted approaches]

.. [PythonDocs2024] Python Software Foundation. (2024). *The Python Tutorial*. Retrieved from https://docs.python.org/3/tutorial/ [Standard Python reference for syntax used in TouchDesigner]

.. [NumPyDocs2024] NumPy Developers. (2024). *NumPy Documentation*. Retrieved from https://numpy.org/doc/stable/ [NumPy reference relevant for Script TOP pixel processing]
