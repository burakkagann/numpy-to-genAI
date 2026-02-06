.. _module-4-3-1-plant-generation:

================================
4.3.1 Plant Generation
================================

:Duration: 25-30 minutes
:Level: Intermediate

Overview
========

Lindenmayer systems, commonly called L-systems, provide a powerful framework for generating complex organic structures from simple rules. Originally developed by biologist Aristid Lindenmayer in 1968 to model plant cell growth, these systems have become fundamental tools in computer graphics and generative art [Lindenmayer1968]_.

In this exercise, you will implement an L-system that generates realistic plant structures. By combining string rewriting rules with turtle graphics interpretation, you will discover how a handful of symbols and transformation rules can produce intricate branching patterns that closely resemble natural vegetation.

Learning Objectives
-------------------

By the end of this exercise, you will be able to:

* Understand L-systems as parallel string-rewriting grammars
* Implement production rules that transform axioms into complex instruction strings
* Use turtle graphics to interpret L-system strings as visual drawings
* Create variations of plant structures by modifying rules and parameters


Quick Start: See It In Action
=============================

Run this code to generate your first L-system plant:

.. code-block:: python
   :caption: Generate a fractal plant using L-systems
   :linenos:

   from PIL import Image, ImageDraw
   import math

   # L-system configuration
   axiom = "F"
   rules = {"F": "F[+F]F[-F]F"}
   iterations = 4
   angle = 25.7

   # Generate instruction string
   instructions = axiom
   for _ in range(iterations):
       next_str = ""
       for char in instructions:
           next_str += rules.get(char, char)
       instructions = next_str

   # Draw using turtle graphics
   image = Image.new("RGB", (800, 600), (10, 20, 30))
   draw = ImageDraw.Draw(image)
   x, y = 400, 550
   current_angle = -math.pi / 2
   stack = []

   for symbol in instructions:
       if symbol == "F":
           new_x = x + 5 * math.cos(current_angle)
           new_y = y + 5 * math.sin(current_angle)
           draw.line([(x, y), (new_x, new_y)], fill=(100, 180, 100))
           x, y = new_x, new_y
       elif symbol == "+":
           current_angle -= math.radians(angle)
       elif symbol == "-":
           current_angle += math.radians(angle)
       elif symbol == "[":
           stack.append((x, y, current_angle))
       elif symbol == "]":
           x, y, current_angle = stack.pop()

   image.save("plant_basic.png")

.. figure:: plant_basic.png
   :width: 500px
   :align: center
   :alt: L-system generated plant showing branching green structure against dark background

   The resulting plant structure at iteration depth 4. Notice how simple rules create complex, organic-looking branching patterns.

You just created a plant using an L-system. The algorithm works by repeatedly replacing each character in a string according to production rules, then interpreting the final string as drawing commands. Each iteration adds more branches, creating the self-similar structure characteristic of natural plants.


Core Concepts
=============

Concept 1: What Are L-Systems?
------------------------------

An **L-system** (Lindenmayer system) is a parallel string-rewriting system that consists of three components [Prusinkiewicz1990]_:

1. **Alphabet**: A set of symbols that can be used in strings
2. **Axiom**: The initial string from which generation begins
3. **Production Rules**: Rules that define how each symbol is replaced in each iteration

Unlike traditional sequential rewriting, L-systems apply all rules simultaneously to every symbol in the string. This parallel nature distinguishes them from other grammar systems and makes them particularly suited for modeling biological growth processes.

Consider the production rule ``F → F[+F]F[-F]F``. When applied to the axiom ``F``, we get:

* **Iteration 0**: ``F`` (1 character)
* **Iteration 1**: ``F[+F]F[-F]F`` (11 characters)
* **Iteration 2**: ``F[+F]F[-F]F[+F[+F]F[-F]F]F[+F]F[-F]F[-F[+F]F[-F]F]F[+F]F[-F]F`` (61 characters)

Each F in the string becomes five new F symbols plus branching markers, causing exponential growth. By iteration 4, the string contains over 1,500 characters [Smith1984]_.

.. admonition:: Did You Know?

   Aristid Lindenmayer originally developed L-systems not for computer graphics but to model the growth patterns of algae and simple plants. His 1968 paper described how the same mathematical framework could capture both the multicellular filaments of blue-green bacteria and the branching patterns of higher plants [Lindenmayer1968]_.


Concept 2: Production Rules and String Rewriting
------------------------------------------------

Production rules define the transformation from one generation to the next. Each rule has two parts:

* **Predecessor**: The symbol to be replaced
* **Successor**: The string that replaces it

For plant generation, we use rules that create branching structures:

.. code-block:: python
   :caption: Common plant generation rules

   # Rule set for a simple weed-like plant
   rules = {"F": "F[+F]F[-F]F"}

   # Rule set for a bushier plant
   rules = {"F": "FF+[+F-F-F]-[-F+F+F]"}

   # Rule set with separate branching
   rules = {"F": "F[+F]F[-F][F]"}

The rule ``F → F[+F]F[-F]F`` can be understood as: "grow forward, sprout a left branch, grow forward, sprout a right branch, grow forward." This creates the characteristic alternating branching pattern seen in many plants.

Here is how string rewriting works in practice:

.. code-block:: python
   :caption: String rewriting implementation

   def apply_rules(axiom, rules, iterations):
       """Apply production rules for n iterations."""
       current = axiom

       for iteration in range(iterations):
           next_string = ""
           for character in current:
               # Replace character if rule exists, otherwise keep it
               next_string += rules.get(character, character)
           current = next_string

       return current

   # Example usage
   axiom = "F"
   rules = {"F": "F[+F]F[-F]F"}
   result = apply_rules(axiom, rules, 2)
   print(f"Result has {len(result)} characters")

The key insight is that all replacements happen simultaneously. Every F in the current string becomes the successor string in a single step, which differs from sequential processing where changes would compound within a single iteration.


Concept 3: Turtle Graphics Interpretation
-----------------------------------------

After generating the L-system string, we need to convert it into a visual image. This is done using **turtle graphics**, a method where an imaginary turtle walks across the canvas following instructions [Abelson1986]_.

Each symbol in the L-system string maps to a turtle command:

.. list-table:: Symbol Meanings
   :widths: 15 85
   :header-rows: 1

   * - Symbol
     - Action
   * - F
     - Move forward by step size and draw a line
   * - \+
     - Turn left (counterclockwise) by the angle
   * - \-
     - Turn right (clockwise) by the angle
   * - [
     - Save current position and angle to stack (push)
   * - ]
     - Restore saved position and angle from stack (pop)

The bracket symbols ``[`` and ``]`` enable branching. When the turtle encounters ``[``, it saves its current state. It can then wander off to draw a branch, and when it reaches ``]``, it returns to the saved position to continue with the main stem.

.. code-block:: python
   :caption: Turtle graphics implementation

   def draw_lsystem(instructions, angle_degrees, step_size, width, height):
       """Convert L-system string to image using turtle graphics."""
       image = Image.new("RGB", (width, height), (10, 20, 30))
       draw = ImageDraw.Draw(image)

       # Starting position and direction
       x, y = width // 2, height - 50
       current_angle = -math.pi / 2  # Point upward

       # Stack for saving turtle state
       state_stack = []

       for symbol in instructions:
           if symbol == "F":
               new_x = x + step_size * math.cos(current_angle)
               new_y = y + step_size * math.sin(current_angle)
               draw.line([(x, y), (new_x, new_y)], fill=(100, 180, 100))
               x, y = new_x, new_y
           elif symbol == "+":
               current_angle -= math.radians(angle_degrees)
           elif symbol == "-":
               current_angle += math.radians(angle_degrees)
           elif symbol == "[":
               state_stack.append((x, y, current_angle))
           elif symbol == "]":
               x, y, current_angle = state_stack.pop()

       return image

.. figure:: lsystem_diagram.png
   :width: 600px
   :align: center
   :alt: Diagram showing L-system string rewriting process and symbol meanings

   Visual explanation of how L-system rules transform strings and how symbols map to turtle commands. Diagram generated with Claude Code.


Hands-On Exercises
==================

Exercise 1: Execute and Explore
-------------------------------

Run the main plant generation script and observe how the output changes with different iteration depths:

.. code-block:: python
   :caption: plant_lsystem.py

   from PIL import Image, ImageDraw
   import math

   # L-System configuration
   AXIOM = "F"
   RULES = {"F": "F[+F]F[-F]F"}
   ANGLE = 25.7

   def apply_rules(axiom, rules, iterations):
       current = axiom
       for _ in range(iterations):
           next_str = ""
           for char in current:
               next_str += rules.get(char, char)
           current = next_str
       return current

   def draw_plant(instructions, angle, step_size, width=800, height=600):
       image = Image.new("RGB", (width, height), (10, 20, 30))
       draw = ImageDraw.Draw(image)

       x, y = width // 2, height - 50
       current_angle = -math.pi / 2
       stack = []

       for symbol in instructions:
           if symbol == "F":
               new_x = x + step_size * math.cos(current_angle)
               new_y = y + step_size * math.sin(current_angle)
               draw.line([(x, y), (new_x, new_y)], fill=(100, 180, 100))
               x, y = new_x, new_y
           elif symbol == "+":
               current_angle -= math.radians(angle)
           elif symbol == "-":
               current_angle += math.radians(angle)
           elif symbol == "[":
               stack.append((x, y, current_angle))
           elif symbol == "]":
               x, y, current_angle = stack.pop()

       return image

   # Generate and save
   instructions = apply_rules(AXIOM, RULES, 4)
   plant = draw_plant(instructions, ANGLE, step_size=5)
   plant.save("my_plant.png")

After running the code, answer these reflection questions:

1. How does the plant change from iteration 1 to iteration 4?
2. What causes the branching effect in the visualization?
3. Why does the instruction string grow exponentially with each iteration?

.. dropdown:: Answers and Explanation
   :class-title: sd-font-weight-bold

   1. **Iteration changes**: At iteration 1, you see a simple Y-shaped plant. By iteration 4, the structure has hundreds of branches creating a dense, bush-like appearance. Each iteration adds more detail to every existing branch.

   2. **Branching effect**: The bracket symbols ``[`` and ``]`` create branches. When ``[`` is encountered, the turtle saves its position. It draws a branch, then ``]`` returns it to the saved position to continue the main stem. This allows multiple paths from a single point.

   3. **Exponential growth**: Each F in the string becomes ``F[+F]F[-F]F`` (5 new Fs plus brackets). If you have N characters containing F symbols, the next iteration will have roughly 5N characters. This creates exponential growth: 1 → 11 → 61 → 331 → 1561 characters.


Exercise 2: Modify Parameters
-----------------------------

Experiment with the L-system by changing parameters to create different plant shapes.

**Goal 1**: Change the branching angle from 25.7 degrees to 45 degrees

Modify the angle constant and observe how it affects the plant structure:

.. code-block:: python

   # Original angle (creates natural-looking plants)
   ANGLE = 25.7

   # Wider angle (creates more spread-out branches)
   ANGLE = 45

   # Narrow angle (creates tight, columnar plants)
   ANGLE = 15

.. dropdown:: Solution: Angle Effects
   :class-title: sd-font-weight-bold

   * **15 degrees**: Creates tall, narrow plants that resemble cypress trees
   * **25.7 degrees**: The golden angle approximation creates natural-looking branching
   * **45 degrees**: Creates wide, spreading plants like oak branches
   * **90 degrees**: Creates geometric, cross-shaped patterns

**Goal 2**: Modify the production rule to create denser branching

Try different rule sets to see how they affect the output:

.. code-block:: python

   # Original rule (simple alternating branches)
   RULES = {"F": "F[+F]F[-F]F"}

   # Bushier plant (more branches per segment)
   RULES = {"F": "FF+[+F-F-F]-[-F+F+F]"}

   # Three-way branching
   RULES = {"F": "F[+F][-F]F"}

.. dropdown:: Solution: Rule Effects
   :class-title: sd-font-weight-bold

   * The original rule creates alternating left-right branching
   * The bushier rule creates multiple sub-branches at each node
   * Three-way branching removes the middle segment, creating cleaner forks

**Goal 3**: Adjust the step size to change plant scale

.. code-block:: python

   # Large steps (sparse, tall plant)
   plant = draw_plant(instructions, ANGLE, step_size=10)

   # Small steps (dense, compact plant)
   plant = draw_plant(instructions, ANGLE, step_size=3)


Exercise 3: Create Custom Plant
-------------------------------

Design your own L-system plant by defining custom axiom and rules.

**Requirements**:

* Create a plant with at least 3 visible branching levels
* Use a different production rule than the examples
* Choose an angle that creates pleasing proportions

**Starter Code**:

.. code-block:: python
   :caption: custom_plant.py (complete the TODO sections)
   :linenos:

   from PIL import Image, ImageDraw
   import math

   # TODO: Define your custom L-system
   AXIOM = "X"  # Starting symbol
   RULES = {
       # TODO: Define rules for X and F
       "X": "...",  # Main growth rule
       "F": "..."   # How stems grow
   }
   ANGLE = ...  # TODO: Choose your angle
   ITERATIONS = ...  # TODO: Choose iteration count

   def apply_rules(axiom, rules, iterations):
       current = axiom
       for _ in range(iterations):
           next_str = ""
           for char in current:
               next_str += rules.get(char, char)
           current = next_str
       return current

   def draw_plant(instructions, angle, step_size=5):
       image = Image.new("RGB", (800, 600), (10, 20, 30))
       draw = ImageDraw.Draw(image)
       x, y = 400, 550
       current_angle = -math.pi / 2
       stack = []

       for symbol in instructions:
           if symbol == "F":
               new_x = x + step_size * math.cos(current_angle)
               new_y = y + step_size * math.sin(current_angle)
               draw.line([(x, y), (new_x, new_y)], fill=(100, 180, 100))
               x, y = new_x, new_y
           elif symbol == "+":
               current_angle -= math.radians(angle)
           elif symbol == "-":
               current_angle += math.radians(angle)
           elif symbol == "[":
               stack.append((x, y, current_angle))
           elif symbol == "]":
               x, y, current_angle = stack.pop()

       return image

   instructions = apply_rules(AXIOM, RULES, ITERATIONS)
   plant = draw_plant(instructions, ANGLE)
   plant.save("custom_plant.png")

.. dropdown:: Hint: Two-symbol Systems
   :class-title: sd-font-weight-bold

   Using two symbols (like X and F) allows more control. X can define the branching structure while F handles actual drawing:

   .. code-block:: python

      RULES = {
          "X": "F+[[X]-X]-F[-FX]+X",  # Branching structure
          "F": "FF"                    # Stem growth
      }

.. dropdown:: Complete Solution
   :class-title: sd-font-weight-bold

   .. code-block:: python
      :linenos:

      from PIL import Image, ImageDraw
      import math

      # A fern-like plant with two symbols
      AXIOM = "X"
      RULES = {
          "X": "F+[[X]-X]-F[-FX]+X",
          "F": "FF"
      }
      ANGLE = 25
      ITERATIONS = 5

      def apply_rules(axiom, rules, iterations):
          current = axiom
          for _ in range(iterations):
              next_str = ""
              for char in current:
                  next_str += rules.get(char, char)
              current = next_str
          return current

      def draw_plant(instructions, angle, step_size=3):
          image = Image.new("RGB", (800, 600), (10, 20, 30))
          draw = ImageDraw.Draw(image)
          x, y = 150, 550
          current_angle = -math.pi / 2
          stack = []

          for symbol in instructions:
              if symbol == "F":
                  new_x = x + step_size * math.cos(current_angle)
                  new_y = y + step_size * math.sin(current_angle)
                  draw.line([(x, y), (new_x, new_y)], fill=(80, 160, 80))
                  x, y = new_x, new_y
              elif symbol == "+":
                  current_angle -= math.radians(angle)
              elif symbol == "-":
                  current_angle += math.radians(angle)
              elif symbol == "[":
                  stack.append((x, y, current_angle))
              elif symbol == "]":
                  x, y, current_angle = stack.pop()

          return image

      instructions = apply_rules(AXIOM, RULES, ITERATIONS)
      plant = draw_plant(instructions, ANGLE)
      plant.save("fern_plant.png")

**Challenge Extension**: Add color variation based on branch depth. Track the stack depth and use it to change line colors from brown (stems) to green (leaves):

.. dropdown:: Challenge Solution
   :class-title: sd-font-weight-bold

   .. code-block:: python

      def draw_plant_with_depth_color(instructions, angle, step_size=3):
          image = Image.new("RGB", (800, 600), (10, 20, 30))
          draw = ImageDraw.Draw(image)
          x, y = 400, 550
          current_angle = -math.pi / 2
          stack = []
          depth = 0

          for symbol in instructions:
              if symbol == "F":
                  # Color varies from brown to green based on depth
                  ratio = min(depth / 6, 1.0)
                  r = int(139 * (1 - ratio) + 50 * ratio)
                  g = int(69 * (1 - ratio) + 180 * ratio)
                  b = int(19 * (1 - ratio) + 50 * ratio)

                  new_x = x + step_size * math.cos(current_angle)
                  new_y = y + step_size * math.sin(current_angle)
                  draw.line([(x, y), (new_x, new_y)], fill=(r, g, b))
                  x, y = new_x, new_y
              elif symbol == "+":
                  current_angle -= math.radians(angle)
              elif symbol == "-":
                  current_angle += math.radians(angle)
              elif symbol == "[":
                  stack.append((x, y, current_angle))
                  depth += 1
              elif symbol == "]":
                  x, y, current_angle = stack.pop()
                  depth -= 1

          return image


Summary
=======

Key Takeaways
-------------

* **L-systems** are parallel string-rewriting grammars that generate complex patterns from simple rules
* The **axiom** is the starting string, and **production rules** define character replacements
* **Turtle graphics** interprets symbols as movement and drawing commands
* **Bracket symbols** ``[`` and ``]`` enable branching by saving and restoring turtle state
* The **branching angle** dramatically affects the visual appearance of generated plants
* L-systems model natural growth processes, explaining why the results look organic

Common Pitfalls
---------------

* **Stack underflow**: Ensure every ``]`` has a matching ``[`` in your rules
* **Runaway growth**: Too many iterations can create millions of characters; start with 3-4 iterations
* **Tiny plants**: Adjust step size and canvas size to fit the generated structure
* **Inverted angles**: Remember that screen coordinates have Y increasing downward
* **Missing symbols**: Rules that do not include all symbols may lose information

Connection to Future Learning
-----------------------------

This exercise establishes foundations for more advanced L-system topics:

* **Module 4.3.2 Koch Snowflake**: Using L-systems to generate geometric fractals
* **Module 4.3.3 Penrose Tiling**: Aperiodic tilings defined by L-system rules
* **Module 4.2.1 Fractal Trees**: Natural tree structures with stochastic variation


Next Steps
==========

Continue to :doc:`../4.3.2_koch_snowflake/README` to explore how L-systems can generate classic geometric fractals like the Koch Snowflake.


References
==========

.. [Lindenmayer1968] Lindenmayer, A. (1968). Mathematical models for cellular interactions in development I. Filaments with one-sided inputs. *Journal of Theoretical Biology*, 18(3), 280-299. https://doi.org/10.1016/0022-5193(68)90079-9

.. [Prusinkiewicz1990] Prusinkiewicz, P., & Lindenmayer, A. (1990). *The Algorithmic Beauty of Plants*. Springer-Verlag. ISBN: 978-0-387-97297-8

.. [Smith1984] Smith, A. R. (1984). Plants, fractals, and formal languages. *ACM SIGGRAPH Computer Graphics*, 18(3), 1-10. https://doi.org/10.1145/964965.808571

.. [Abelson1986] Abelson, H., & diSessa, A. A. (1986). *Turtle Geometry: The Computer as a Medium for Exploring Mathematics*. MIT Press. ISBN: 978-0-262-51037-0

.. [Prusinkiewicz2012] Prusinkiewicz, P., & Runions, A. (2012). Computational models of plant development and form. *New Phytologist*, 193(3), 549-569. https://doi.org/10.1111/j.1469-8137.2011.04009.x

.. [Rozenberg1980] Rozenberg, G., & Salomaa, A. (1980). *The Mathematical Theory of L Systems*. Academic Press. ISBN: 978-0-12-597140-8

.. [Mandelbrot1982] Mandelbrot, B. B. (1982). *The Fractal Geometry of Nature*. W. H. Freeman and Company. ISBN: 978-0-7167-1186-5

.. [NumPyDocs] NumPy Developers. (2024). NumPy array indexing. *NumPy Documentation*. https://numpy.org/doc/stable/user/basics.indexing.html

.. [PillowDocs] Clark, A., et al. (2024). *Pillow: Python Imaging Library* (Version 10.2.0). Python Software Foundation. https://pillow.readthedocs.io/
