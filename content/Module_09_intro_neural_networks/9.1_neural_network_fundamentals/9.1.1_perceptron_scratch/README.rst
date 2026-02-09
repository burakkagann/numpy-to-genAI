.. _module-9-1-1-perceptron-scratch:

================================
9.1.1 Perceptron from Scratch
================================

:Duration: 35-40 minutes
:Level: Intermediate

Overview
========

Rosenblatt built the first perceptron in 1958 using three operations: multiply inputs by weights, sum the products, and check if the result crosses a threshold [LeCun2015a]_. We will build a working binary classifier in about 30 lines of NumPy. No frameworks, no hidden magic.

Frank Rosenblatt introduced the perceptron in 1958 as a computational model inspired by biological neurons [Rosenblatt1958]_. It was the first machine that could learn to classify data [Crevier1993]_. The same weight-update rule still runs inside networks with billions of parameters today. When you write the weight update yourself, you see exactly where learning happens: right after each wrong prediction.

Learning Objectives
-------------------

By the end of this exercise, you will be able to:

* Understand perceptron architecture with inputs, weights, bias, and activation
* Implement the forward pass to compute predictions from inputs
* Apply the perceptron learning rule to update weights based on errors
* Visualize decision boundaries learned by the perceptron


Quick Start: See It In Action
=============================

Download and run the script to train a perceptron and visualize its learned decision boundary:

:download:`Download simple_perceptron.py <simple_perceptron.py>`

.. code-block:: bash

   python simple_perceptron.py

.. figure:: perceptron_output.png
   :width: 450px
   :align: center
   :alt: Decision boundary learned by perceptron showing blue and orange regions separated by a line, with data points colored by class

   The trained perceptron divides the feature space into two regions.

The perceptron learns by adjusting its weights whenever it makes an error. After training, the weights define a linear boundary that separates the two classes. If a straight line can separate the two classes, this learning rule will find it. Novikoff proved that in 1962.


Core Concepts
=============

Core Concept 1: What is a Perceptron?
--------------------------------------

Imagine taking two numbers, multiplying each by a weight, adding them up with a bias term, and checking whether the total is positive or negative. That check decides the output: 1 or 0. This is what a **perceptron** does. It was modeled on a simplified view of biological neurons [McCulloch1943]_, where dendrites receive signals, the cell body combines them, and the axon fires if the total exceeds a threshold.

The perceptron takes multiple numerical inputs, multiplies each by a corresponding weight, adds them together with a bias term, and applies an activation function to produce a binary output (0 or 1). Mathematically:

.. code-block:: text

   output = step(w1*x1 + w2*x2 + ... + wn*xn + b)

Where:

* **x1, x2, ..., xn** are the input features
* **w1, w2, ..., wn** are the weights (learned from data)
* **b** is the bias term (also learned)
* **step(z)** is the step function: returns 1 if z >= 0, else 0

.. figure:: perceptron_architecture.png
   :width: 550px
   :align: center
   :alt: Diagram showing perceptron architecture with input nodes x1 and x2, weights w1 and w2, summation node, step activation, and output y

   Perceptron architecture: inputs, weights, summation, and step activation. Diagram generated with Claude - Opus 4.5.

The weights determine how much each input contributes to the output. A large positive weight means that input strongly influences the output toward 1, while a large negative weight pushes it toward 0. The bias acts like a threshold, shifting when the perceptron fires.

.. admonition:: Historical Note

   In 1969, Minsky and Papert proved that a single perceptron cannot learn the XOR function [Minsky1969]_. This limitation contributed to the first "AI winter" when funding for neural network research dried up [Crevier1993]_. It took multi-layer networks (with hidden layers) to overcome this limitation, but those required new training algorithms like backpropagation.


Core Concept 2: The Forward Pass
---------------------------------

The **forward pass** is how the perceptron makes predictions. Given an input, it computes the weighted sum of inputs plus bias, then applies the step activation function to produce output.

.. code-block:: python
   :caption: Forward pass implementation
   :linenos:
   :emphasize-lines: 6,9

   def forward(self, x):
       """
       Compute the perceptron output for input x.
       """
       # Step 1: Compute weighted sum (dot product + bias)
       # Formula: z = w1*x1 + w2*x2 + ... + wn*xn + bias
       weighted_sum = np.dot(self.weights, x) + self.bias

       # Step 2: Apply step activation function
       if weighted_sum >= 0:
           return 1
       else:
           return 0

**Line 6** computes the weighted sum using NumPy's dot product [NumPyDocs911]_. This is the linear combination of inputs and weights: w1*x1 + w2*x2 + ... + b.

**Line 9** applies the step function. If the weighted sum is non-negative, the perceptron outputs 1 (predicts Class 1). Otherwise, it outputs 0 (predicts Class 0).

The forward pass is essentially drawing a line through the feature space. Points on one side of the line are Class 0; points on the other side are Class 1. The weights determine the slope of this line, and the bias determines where it crosses.

.. figure:: linearly_separable.png
   :width: 400px
   :align: center
   :alt: Scatter plot showing two clusters of points, orange in lower-left and blue in upper-right, that can be separated by a straight line

   Two clusters separable by a straight line.


Core Concept 3: Learning from Errors
-------------------------------------

The perceptron learns through the **perceptron learning rule**, one of the earliest machine learning algorithms [Rosenblatt1958]_. The key insight is simple: if the prediction is wrong, adjust the weights to make it less wrong next time.

.. code-block:: python
   :caption: Perceptron learning rule
   :linenos:
   :emphasize-lines: 10,11

   def train(self, X, y, epochs=100):
       for epoch in range(epochs):
           for i in range(len(X)):
               # Make prediction
               y_pred = self.forward(X[i])

               # Calculate error
               error = y[i] - y_pred  # +1, 0, or -1

               # Update if wrong
               if error != 0:
                   self.weights = self.weights + self.learning_rate * error * X[i]
                   self.bias = self.bias + self.learning_rate * error

The learning rule states:

* **If prediction is correct** (error = 0): do nothing
* **If predicted 0 but should be 1** (error = +1): increase weights in direction of input
* **If predicted 1 but should be 0** (error = -1): decrease weights in direction of input

The **learning rate** controls how much the weights change on each update. A larger learning rate means bigger steps (faster learning but potentially unstable), while a smaller rate means smaller, more cautious updates.

.. figure:: training_progression.gif
   :width: 400px
   :align: center
   :alt: Animation showing decision boundary rotating and shifting as the perceptron trains over multiple epochs

   Watch the decision boundary evolve during training. Initially random, it gradually rotates to separate the two classes correctly.

.. figure:: concept3_comparison.png
   :width: 700px
   :align: center
   :alt: Side-by-side comparison showing best decision boundary on left and training error progression chart on right

   Best decision boundary (left) and error over epochs (right). Diagram generated with Claude - Opus 4.5.

.. admonition:: Why Does the Error Oscillate?
   :class: note

   You may notice the error count fluctuates rather than decreasing smoothly. This is a fundamental property of the basic perceptron algorithm:

   1. **Online Learning**: The perceptron updates weights after each misclassified point, not after seeing all data. Fixing one point may break another.

   2. **Perceptron Cycling Theorem**: When data is **not** linearly separable (as with our deliberate outliers), the perceptron will eventually cycle through the same weight configurations forever [Block1962]_. It cannot converge because no single line can correctly classify all points.

   3. **No Memory of Best Solution**: The basic perceptron does not track which weights performed best. It simply keeps updating.

   This limitation led to the **Pocket Algorithm** [Gallant1990]_, which keeps the best-performing weights "in its pocket" while continuing to train. Modern neural networks address this through batch learning, early stopping, and saving model checkpoints.

**Convergence Theorem**: If the data is linearly separable, the perceptron learning rule is guaranteed to find a separating boundary in a finite number of steps [Haykin2009]_. This result, formally proven by Albert Novikoff in 1962, was one of the first convergence proofs in machine learning [Novikoff1963]_.


Hands-On Exercises
==================

Now it is time to apply what you've learned with three progressively challenging exercises. Each builds on the previous one using the **Execute → Modify → Create** approach [Sweller1985]_, [Mayer2020]_.

Exercise 1: Execute and Explore
-------------------------------

Download and run the perceptron implementation to observe its behavior.

:download:`Download simple_perceptron.py <simple_perceptron.py>`

After running the code, answer these reflection questions:

1. How many epochs did it take to converge (reach zero errors)?
2. What do the final weights and bias represent geometrically?
3. If you run the training again with a different random seed, do you get the same weights?

.. dropdown:: Answers and Explanation
   :class-title: sd-font-weight-bold

   1. **Epochs to converge**: With seed 42 and this data, the perceptron typically converges in 1-3 epochs. The data is well-separated, making it an easy classification problem.

   2. **Geometric interpretation**: The weights [w1, w2] define the normal vector to the decision boundary line. The equation w1*x1 + w2*x2 + b = 0 describes a line in 2D space. Points where this sum is positive are Class 1; points where it is negative are Class 0.

   3. **Different runs**: With different random seeds (for weight initialization), you will get different final weights. There are infinitely many valid decision boundaries for linearly separable data. The perceptron finds one that works, but not necessarily the same one each time.


Exercise 2: Modify Parameters
-----------------------------

Experiment with the learning rate to understand its effect on training.

**Goal 1**: Test different learning rates

.. code-block:: python
   :caption: Try different learning rates

   # Very slow learning
   p1 = Perceptron(input_size=2, learning_rate=0.01)
   h1 = p1.train(X, y, epochs=50)
   print(f"LR=0.01: {len(h1)} epochs, errors={h1}")

   # Moderate learning
   p2 = Perceptron(input_size=2, learning_rate=0.1)
   h2 = p2.train(X, y, epochs=50)
   print(f"LR=0.1: {len(h2)} epochs")

   # Fast learning
   p3 = Perceptron(input_size=2, learning_rate=1.0)
   h3 = p3.train(X, y, epochs=50)
   print(f"LR=1.0: {len(h3)} epochs")

.. figure:: learning_rate_comparison.png
   :width: 550px
   :align: center
   :alt: Four panels showing decision boundaries for different learning rates, from slow (0.01) to very fast (1.0)

   Decision boundaries for different learning rates. Diagram generated with Claude - Opus 4.5.

**Goal 2**: Modify the data distribution

Try changing the cluster centers to make the problem harder:

.. code-block:: python
   :caption: Closer clusters (harder problem)

   # Closer clusters - harder to separate
   class_0 = np.random.randn(50, 2) * 0.5 + [-0.5, -0.5]
   class_1 = np.random.randn(50, 2) * 0.5 + [0.5, 0.5]

Observe how the number of training epochs changes when the clusters are closer together.

.. dropdown:: Hints and Solutions
   :class-title: sd-font-weight-bold

   **Observations**:

   * **Small learning rate (0.01)**: More epochs needed, but stable convergence
   * **Moderate learning rate (0.1)**: Good balance of speed and stability
   * **Large learning rate (1.0)**: Fewer epochs but can overshoot (not visible for this simple problem)
   * **Closer clusters**: More epochs needed because data points are closer to the boundary


Exercise 3: Create Your Own
---------------------------

Complete the starter code to implement your own perceptron from scratch.

:download:`Download perceptron_starter.py <perceptron_starter.py>`

**Your tasks**:

1. Complete ``__init__`` to initialize weights, bias, and learning rate
2. Complete ``forward`` to compute predictions using the step function
3. Complete ``train`` to update weights using the perceptron learning rule

.. dropdown:: Hint 1: Initialization
   :class-title: sd-font-weight-bold

   For ``__init__``:

   .. code-block:: python

      self.weights = np.random.randn(input_size) * 0.01
      self.bias = 0.0
      self.learning_rate = learning_rate

.. dropdown:: Hint 2: Forward Pass
   :class-title: sd-font-weight-bold

   For ``forward``:

   .. code-block:: python

      weighted_sum = np.dot(self.weights, x) + self.bias
      if weighted_sum >= 0:
          return 1
      else:
          return 0

.. dropdown:: Hint 3: Training Loop
   :class-title: sd-font-weight-bold

   For ``train``:

   .. code-block:: python

      y_pred = self.forward(X[i])
      error = y[i] - y_pred
      if error != 0:
          self.weights = self.weights + self.learning_rate * error * X[i]
          self.bias = self.bias + self.learning_rate * error
          errors += 1

.. dropdown:: Complete Solution
   :class-title: sd-font-weight-bold

   :download:`Download perceptron_solution.py <perceptron_solution.py>`

   .. code-block:: python
      :linenos:

      import numpy as np

      class Perceptron:
          def __init__(self, input_size, learning_rate=0.1):
              self.weights = np.random.randn(input_size) * 0.01
              self.bias = 0.0
              self.learning_rate = learning_rate

          def forward(self, x):
              weighted_sum = np.dot(self.weights, x) + self.bias
              return 1 if weighted_sum >= 0 else 0

          def train(self, X, y, epochs=100):
              for epoch in range(epochs):
                  errors = 0
                  for i in range(len(X)):
                      y_pred = self.forward(X[i])
                      error = y[i] - y_pred
                      if error != 0:
                          self.weights += self.learning_rate * error * X[i]
                          self.bias += self.learning_rate * error
                          errors += 1
                  if errors == 0:
                      print(f"Converged in {epoch + 1} epochs")
                      return
              print(f"Training completed after {epochs} epochs")


Creative Challenge: Geometric Art with Perceptrons
--------------------------------------------------

Now that you understand how perceptrons create decision boundaries, use multiple perceptrons to create abstract geometric art. Each perceptron divides the canvas with a linear boundary, and combining several creates Mondrian-like compositions.

.. figure:: perceptron_art.png
   :width: 400px
   :align: center
   :alt: Abstract geometric art created by 4 perceptrons dividing a canvas into 16 colored regions

   Geometric art generated by 4 perceptrons. Each perceptron creates a linear boundary, resulting in 2^4 = 16 distinct regions with unique colors.

**Your Goal**: Create this image using multiple perceptrons. Try to figure it out yourself before looking at the hints.

.. dropdown:: Hint 1: The Core Idea
   :class-title: sd-font-weight-bold

   Each perceptron outputs 0 or 1 for any point. With 4 perceptrons, each pixel
   gets a 4-bit "signature" (0000 to 1111), creating 16 unique regions. Each
   region gets a different color.

.. dropdown:: Hint 2: Getting Started
   :class-title: sd-font-weight-bold

   1. Create a simple Perceptron class with random weights (no training needed)
   2. Create 4 perceptrons: ``perceptrons = [Perceptron(2) for _ in range(4)]``
   3. For each pixel, compute its signature by calling ``forward()`` on each perceptron

.. dropdown:: Hint 3: Computing the Signature
   :class-title: sd-font-weight-bold

   .. code-block:: python

      # For each pixel at (x, y), normalize to range (-2, 2)
      point = np.array([(x - 200) / 100, (y - 200) / 100])

      # Compute binary signature: sum of 2^i * perceptron_i(point)
      signature = sum(p.forward(point) * (2 ** i)
                      for i, p in enumerate(perceptrons))

.. dropdown:: Hint 4: Assigning Colors
   :class-title: sd-font-weight-bold

   Create a color palette with 16 colors (one per region):

   .. code-block:: python

      colors = [[80 + 140 * ((i * 23 % 16) % 4) // 3,
                 80 + 140 * (((i * 23 % 16) // 2) % 4) // 3,
                 80 + 140 * (((i * 23 % 16) // 4) % 4) // 3]
                for i in range(16)]

:download:`Download Solution <geometric_art_solution.py>`

The solution uses the Pillow library [PillowDocs911]_ to save the generated image.

**Experiments to try**:

* Change the number of perceptrons (3-6 work well)
* Modify the seed for different compositions
* Create your own color palette
* Add gradients within regions based on distance to boundaries

.. note:: Implementation Note

   The perceptron implementations in this exercise are inspired by the
   following foundational references:

   - Rosenblatt, F. (1958). *The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain*, Psychological Review — the original perceptron algorithm and learning rule
   - scikit-learn Perceptron, https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html — modern reference implementation of the perceptron classifier


Summary
=======

Key Takeaways
-------------

The perceptron taught us one update rule: when wrong, nudge the weights by (error x input x learning rate). This same rule, repeated billions of times across millions of weights, trains modern neural networks [Goodfellow2016a]_:

* A **perceptron** is a single artificial neuron that computes a weighted sum of inputs, adds a bias, and applies a step activation function
* The **forward pass** transforms inputs to outputs: y = step(w dot x + b)
* The **perceptron learning rule** updates weights when predictions are wrong: w = w + lr * error * x
* Perceptrons can only learn **linearly separable** patterns (they cannot solve XOR)
* The **decision boundary** is a hyperplane in the input space defined by the weights and bias
* **Convergence is guaranteed** for linearly separable data in finite time

Common Pitfalls
---------------

* **Not normalizing data**: Large input values can cause learning instability
* **Learning rate too high**: Can cause oscillation and prevent convergence
* **Learning rate too low**: Training takes many epochs
* **Non-linearly separable data**: Perceptron will never converge; need multi-layer networks
* **Forgetting bias term**: The bias allows the decision boundary to shift from the origin



References
==========

.. [Rosenblatt1958] Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain. *Psychological Review*, 65(6), 386-408. https://doi.org/10.1037/h0042519 (`PDF <https://www.ling.upenn.edu/courses/cogs501/Rosenblatt1958.pdf>`_)

.. [McCulloch1943] McCulloch, W. S., & Pitts, W. (1943). A logical calculus of the ideas immanent in nervous activity. *Bulletin of Mathematical Biophysics*, 5(4), 115-133. https://doi.org/10.1007/BF02478259

.. [Minsky1969] Minsky, M., & Papert, S. (1969). *Perceptrons: An Introduction to Computational Geometry*. MIT Press. ISBN: 978-0-262-63022-1

.. [Haykin2009] Haykin, S. (2009). *Neural Networks and Learning Machines* (3rd ed.). Pearson. ISBN: 978-0-13-147139-9

.. [Novikoff1963] Novikoff, A. B. J. (1963). On convergence proofs for perceptrons. In *Proceedings of the Symposium on the Mathematical Theory of Automata* (Vol. 12, pp. 615-622). Polytechnic Institute of Brooklyn.

.. [Block1962] Block, H. D. (1962). The perceptron: A model for brain functioning. *Reviews of Modern Physics*, 34(1), 123-135. https://doi.org/10.1103/RevModPhys.34.123

.. [Gallant1990] Gallant, S. I. (1990). Perceptron-based learning algorithms. *IEEE Transactions on Neural Networks*, 1(2), 179-191. https://doi.org/10.1109/72.80230

.. [Crevier1993] Crevier, D. (1993). *AI: The Tumultuous History of the Search for Artificial Intelligence*. Basic Books. ISBN: 978-0-465-02997-6

.. [Goodfellow2016a] Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. ISBN: 978-0-262-03561-3. https://www.deeplearningbook.org/

.. [LeCun2015a] LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436-444. https://doi.org/10.1038/nature14539

.. [NumPyDocs911] NumPy Developers. (2024). NumPy linear algebra (numpy.dot). *NumPy Documentation*. https://numpy.org/doc/stable/reference/generated/numpy.dot.html

.. [PillowDocs911] Clark, A., et al. (2024). *Pillow: Python Imaging Library* (Version 10.2.0). Python Software Foundation. https://pillow.readthedocs.io/

.. [Sweller1985] Sweller, J. (1985). Optimizing cognitive load in instructional design. *Instructional Science*, 14(3), 195-218.

.. [Mayer2020] Mayer, R. E. (2020). *Multimedia Learning* (3rd ed.). Cambridge University Press. ISBN: 978-1-316-63896-8
