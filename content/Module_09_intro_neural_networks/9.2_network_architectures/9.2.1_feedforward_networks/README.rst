.. _module-9-2-1-feedforward-networks:

================================
9.2.1 Feedforward Networks
================================

:Duration: 35-40 minutes
:Level: Intermediate-Advanced

Overview
========

A single perceptron can only learn linear decision boundaries, making it fundamentally limited. In Exercise 9.1.1, we noted that perceptrons cannot solve the XOR problem because no single straight line can separate the classes. **Feedforward networks** overcome this limitation by adding hidden layers between input and output, enabling the learning of complex, non-linear patterns.

In this exercise, you will build a multi-layer neural network from scratch and train it to solve the XOR problem. By visualizing how the decision boundary evolves during training, you will develop intuition for how hidden layers combine simple linear operations to create sophisticated non-linear classifiers. This is the fundamental insight that powers all deep learning.

Learning Objectives
-------------------

By the end of this exercise, you will be able to:

* Explain why multi-layer networks can learn patterns that single perceptrons cannot
* Implement a feedforward network with hidden layers using NumPy
* Apply backpropagation through multiple layers to train the network
* Visualize decision boundaries and understand how they become non-linear


Quick Start: See It In Action
=============================

Run this code to train a feedforward network on the XOR problem:

.. code-block:: python
   :caption: Train a feedforward network to solve XOR
   :linenos:

   import numpy as np
   from PIL import Image, ImageDraw

   def sigmoid(x):
       return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

   def sigmoid_derivative(x):
       s = sigmoid(x)
       return s * (1 - s)

   class FeedforwardNetwork:
       def __init__(self, input_size=2, hidden_size=4, output_size=1, learning_rate=0.5):
           self.learning_rate = learning_rate
           self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.5
           self.bias_hidden = np.zeros((1, hidden_size))
           self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.5
           self.bias_output = np.zeros((1, output_size))

       def forward(self, X):
           self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
           self.hidden_output = sigmoid(self.hidden_input)
           self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
           self.output = sigmoid(self.output_input)
           return self.output

       def backward(self, X, y):
           n_samples = X.shape[0]
           output_error = self.output - y
           output_delta = output_error * sigmoid_derivative(self.output_input)
           hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
           hidden_delta = hidden_error * sigmoid_derivative(self.hidden_input)
           self.weights_hidden_output -= self.learning_rate * np.dot(self.hidden_output.T, output_delta) / n_samples
           self.bias_output -= self.learning_rate * np.mean(output_delta, axis=0, keepdims=True)
           self.weights_input_hidden -= self.learning_rate * np.dot(X.T, hidden_delta) / n_samples
           self.bias_hidden -= self.learning_rate * np.mean(hidden_delta, axis=0, keepdims=True)

       def train(self, X, y, epochs=5000):
           for epoch in range(epochs):
               self.forward(X)
               self.backward(X, y)

   # XOR dataset
   X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
   y = np.array([[0], [1], [1], [0]])

   np.random.seed(42)
   network = FeedforwardNetwork()
   network.train(X, y, epochs=5000)

   print("Predictions:")
   for i in range(len(X)):
       pred = network.forward(X[i:i+1])[0, 0]
       print(f"Input: {X[i]} -> Output: {pred:.4f} (Class {int(round(pred))})")

.. figure:: feedforward_output.png
   :width: 450px
   :align: center
   :alt: Decision boundary learned by feedforward network showing non-linear separation of XOR data with blue and orange regions

   The trained feedforward network creates a non-linear decision boundary that successfully separates the XOR data. Unlike the straight-line boundary of a perceptron, this boundary curves to classify all four points correctly.

The network achieves 100% accuracy on XOR, something a single perceptron can never do. The hidden layer transforms the input space, making the previously inseparable classes separable.


Core Concepts
=============

Concept 1: Why Multi-Layer Networks?
------------------------------------

A **single perceptron** creates a linear decision boundary: a straight line in 2D, a plane in 3D, or a hyperplane in higher dimensions. This means it can only classify data that is **linearly separable**, where a single straight line can divide the classes [Minsky1969]_.

The XOR problem is the classic example of **non-linearly separable** data:

.. figure:: xor_problem.png
   :width: 400px
   :align: center
   :alt: XOR data points showing diagonal corners belong to the same class, making linear separation impossible

   The XOR problem: points at (0,0) and (1,1) are class 0 (blue), while (0,1) and (1,0) are class 1 (orange). No single straight line can separate these classes.

XOR returns 1 when exactly one input is 1, and 0 otherwise. The same-class points are on opposite corners of the unit square, making it impossible to draw a single line between them.

**Hidden layers solve this problem** by transforming the input space. Each hidden neuron computes its own linear boundary. When combined, these boundaries create a non-linear decision region. The **Universal Approximation Theorem** proves that a neural network with just one hidden layer containing enough neurons can approximate any continuous function to arbitrary precision [Hornik1989]_.

.. admonition:: Did You Know?

   In 1969, Minsky and Papert published *Perceptrons*, proving that single-layer networks cannot solve XOR [Minsky1969]_. This contributed to the first "AI Winter" when neural network research funding dried up. It took until the 1980s, when backpropagation for multi-layer networks was popularized by Rumelhart, Hinton, and Williams [Rumelhart1986]_, for the field to recover.


Concept 2: The Architecture of Hidden Layers
---------------------------------------------

A **feedforward network** consists of layers where information flows in one direction: from input to output, with no cycles. The layers are:

* **Input Layer**: Receives the raw data (for XOR: 2 neurons for x1 and x2)
* **Hidden Layer(s)**: Transforms the data through learned weights and activations
* **Output Layer**: Produces the final prediction

.. figure:: network_architecture.png
   :width: 550px
   :align: center
   :alt: Diagram showing feedforward network with 2 input neurons, 4 hidden neurons, and 1 output neuron, connected by weighted edges

   A feedforward network with architecture 2-4-1: two input neurons, four hidden neurons, and one output neuron. Each connection has a learnable weight.

The **forward pass** computes the output step by step:

.. code-block:: python
   :caption: Forward pass through hidden layer
   :linenos:
   :emphasize-lines: 3,6,9,12

   def forward(self, X):
       # Step 1: Compute hidden layer input (weighted sum)
       self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden

       # Step 2: Apply activation function to hidden layer
       self.hidden_output = sigmoid(self.hidden_input)

       # Step 3: Compute output layer input
       self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output

       # Step 4: Apply activation to output
       self.output = sigmoid(self.output_input)

       return self.output

* **Lines 3**: Matrix multiplication transforms input to hidden space
* **Line 6**: Sigmoid activation introduces non-linearity
* **Lines 9, 12**: Same operations for hidden-to-output transformation

The key insight is that the hidden layer creates a **new representation** of the input. In this transformed space, the XOR classes become linearly separable.


Concept 3: Decision Boundaries and Non-Linearity
-------------------------------------------------

Each hidden neuron acts like a perceptron, creating its own linear boundary. The output neuron then combines these boundaries. With four hidden neurons, we have four linear "cuts" that together carve out a non-linear region.

.. figure:: decision_boundary_evolution.gif
   :width: 400px
   :align: center
   :alt: Animation showing the decision boundary evolving from random to correctly separating XOR classes during training

   Watch the decision boundary evolve during training. The boundary starts random and gradually forms the characteristic non-linear shape that correctly classifies all XOR points.

The animation shows how training shapes the boundary:

1. **Initial state**: Random weights create an arbitrary (and wrong) boundary
2. **Early training**: The boundary begins shifting toward the data points
3. **Mid training**: The boundary curves to separate opposite corners
4. **Final state**: A clear non-linear boundary separates all four points correctly

.. figure:: perceptron_vs_feedforward.png
   :width: 600px
   :align: center
   :alt: Side-by-side comparison showing perceptron failing on XOR (linear boundary) versus feedforward network succeeding (curved boundary)

   Left: A single perceptron can only create a straight line, failing on XOR. Right: A feedforward network with hidden layers creates a curved boundary that succeeds.

The hidden layer's role is to "twist" the input space so that points that were inseparable become separable. This is the fundamental power of deep learning: each layer learns increasingly useful representations [LeCun2015]_.


Hands-On Exercises
==================

Exercise 1: Execute and Explore
-------------------------------

Run the complete feedforward network implementation:

.. code-block:: python
   :caption: feedforward_network.py
   :linenos:

   import numpy as np
   from PIL import Image, ImageDraw

   def sigmoid(x):
       return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

   def sigmoid_derivative(x):
       s = sigmoid(x)
       return s * (1 - s)

   class FeedforwardNetwork:
       def __init__(self, input_size=2, hidden_size=4, output_size=1, learning_rate=0.5):
           self.learning_rate = learning_rate
           self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.5
           self.bias_hidden = np.zeros((1, hidden_size))
           self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.5
           self.bias_output = np.zeros((1, output_size))

       def forward(self, X):
           self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
           self.hidden_output = sigmoid(self.hidden_input)
           self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
           self.output = sigmoid(self.output_input)
           return self.output

       def backward(self, X, y):
           n_samples = X.shape[0]
           output_error = self.output - y
           output_delta = output_error * sigmoid_derivative(self.output_input)
           hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
           hidden_delta = hidden_error * sigmoid_derivative(self.hidden_input)
           self.weights_hidden_output -= self.learning_rate * np.dot(self.hidden_output.T, output_delta) / n_samples
           self.bias_output -= self.learning_rate * np.mean(output_delta, axis=0, keepdims=True)
           self.weights_input_hidden -= self.learning_rate * np.dot(X.T, hidden_delta) / n_samples
           self.bias_hidden -= self.learning_rate * np.mean(hidden_delta, axis=0, keepdims=True)

       def train(self, X, y, epochs=5000):
           loss_history = []
           for epoch in range(epochs):
               output = self.forward(X)
               loss = np.mean((y - output) ** 2)
               loss_history.append(loss)
               self.backward(X, y)
               if epoch % 1000 == 0:
                   print(f"Epoch {epoch}: Loss = {loss:.6f}")
           return loss_history

   # XOR dataset
   X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
   y = np.array([[0], [1], [1], [0]])

   np.random.seed(42)
   network = FeedforwardNetwork()
   history = network.train(X, y, epochs=5000)

   print("\nFinal predictions:")
   predictions = network.forward(X)
   for i in range(len(X)):
       print(f"Input: {X[i]} -> {predictions[i,0]:.4f} (Class {int(round(predictions[i,0]))})")

After running the code, answer these reflection questions:

1. How does the loss change during training? Does it decrease steadily?
2. What are the final prediction values? Are they close to 0 and 1, or somewhere in between?
3. Why does the network need thousands of epochs to converge, unlike the perceptron which converged in just a few epochs?

.. dropdown:: Answers and Explanation
   :class-title: sd-font-weight-bold

   1. **Loss trajectory**: The loss starts around 0.25 (random guess) and gradually decreases. Unlike perceptron training which makes discrete jumps, gradient descent creates a smooth descent. Loss may plateau temporarily before dropping further.

   2. **Prediction values**: Final predictions are typically around 0.2-0.3 for class 0 and 0.7-0.8 for class 1, not exactly 0 or 1. This is because sigmoid asymptotically approaches 0 and 1 but never reaches them. The network correctly classifies all points (rounding gives 100% accuracy).

   3. **Training time**: The perceptron uses the simple perceptron learning rule which makes large discrete updates. The feedforward network uses gradient descent with small learning rate steps, requiring many iterations. Additionally, the error must propagate through multiple layers (backpropagation), and the sigmoid's gradients can be small, slowing learning.


Exercise 2: Modify Parameters
-----------------------------

Experiment with network architecture and training parameters.

**Goal 1**: Try different hidden layer sizes

.. code-block:: python
   :caption: Vary hidden layer size

   hidden_sizes = [2, 4, 8, 16]
   for h in hidden_sizes:
       np.random.seed(42)
       network = FeedforwardNetwork(hidden_size=h)
       network.train(X, y, epochs=5000)
       predictions = network.forward(X)
       accuracy = np.mean(np.round(predictions) == y) * 100
       print(f"Hidden size {h}: Accuracy = {accuracy:.1f}%")

.. figure:: depth_comparison.png
   :width: 550px
   :align: center
   :alt: Four panels showing decision boundaries for networks with 2, 4, 8, and 16 hidden neurons

   Comparison of decision boundaries with different hidden layer sizes. More neurons can create more complex boundaries, but 4 neurons are sufficient for XOR.

**Goal 2**: Vary the learning rate

.. code-block:: python
   :caption: Test different learning rates

   learning_rates = [0.1, 0.5, 1.0, 2.0]
   for lr in learning_rates:
       np.random.seed(42)
       network = FeedforwardNetwork(learning_rate=lr)
       history = network.train(X, y, epochs=5000)
       final_loss = history[-1]
       print(f"Learning rate {lr}: Final loss = {final_loss:.6f}")

**Goal 3**: Change the random seed

.. code-block:: python
   :caption: Different weight initializations

   for seed in [1, 42, 123, 999]:
       np.random.seed(seed)
       network = FeedforwardNetwork()
       network.train(X, y, epochs=5000)
       predictions = network.forward(X)
       accuracy = np.mean(np.round(predictions) == y) * 100
       print(f"Seed {seed}: Accuracy = {accuracy:.1f}%")

.. dropdown:: Observations and Insights
   :class-title: sd-font-weight-bold

   **Hidden layer size**:
   - 2 neurons: Often fails or takes longer. XOR needs at least 2 hidden neurons theoretically, but learning is harder.
   - 4 neurons: Reliably solves XOR. This is a good default.
   - 8-16 neurons: Works well but no significant improvement. More neurons add unnecessary parameters.

   **Learning rate**:
   - 0.1: Converges slowly but reliably.
   - 0.5: Good balance of speed and stability.
   - 1.0-2.0: May oscillate or diverge. High learning rates can overshoot the minimum.

   **Random initialization**:
   - Different seeds lead to different final weights but usually similar accuracy.
   - Some seeds may get stuck in local minima. Training longer or using momentum can help.


Exercise 3: Create Your Own
---------------------------

Complete the starter code to implement a feedforward network from scratch.

**Requirements**:

* Initialize weights for input-to-hidden and hidden-to-output connections
* Implement the forward pass through the hidden layer
* Implement backpropagation to update all weights

**Starter Code**:

:download:`Download feedforward_starter.py <feedforward_starter.py>`

.. code-block:: python
   :caption: feedforward_starter.py (complete the TODO sections)
   :linenos:

   import numpy as np

   def sigmoid(x):
       return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

   def sigmoid_derivative(x):
       s = sigmoid(x)
       return s * (1 - s)

   class FeedforwardNetwork:
       def __init__(self, input_size=2, hidden_size=4, output_size=1, learning_rate=0.5):
           self.learning_rate = learning_rate

           # TODO: Initialize weights from input to hidden layer
           # Shape: (input_size, hidden_size)
           self.weights_input_hidden = None  # Replace

           # TODO: Initialize biases for hidden layer
           self.bias_hidden = None  # Replace

           # TODO: Initialize weights from hidden to output
           self.weights_hidden_output = None  # Replace

           # TODO: Initialize bias for output
           self.bias_output = None  # Replace

       def forward(self, X):
           # TODO: Compute hidden layer input and output
           self.hidden_input = None  # Replace
           self.hidden_output = None  # Replace

           # TODO: Compute output layer
           self.output_input = None  # Replace
           self.output = None  # Replace

           return self.output

       def backward(self, X, y):
           n_samples = X.shape[0]

           # TODO: Calculate output error and delta
           output_error = None  # Replace
           output_delta = None  # Replace

           # TODO: Calculate hidden error and delta
           hidden_error = None  # Replace
           hidden_delta = None  # Replace

           # TODO: Update all weights and biases
           pass  # Replace with weight updates

.. dropdown:: Hint 1: Weight Initialization
   :class-title: sd-font-weight-bold

   Use random initialization with small values to break symmetry:

   .. code-block:: python

      self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.5
      self.bias_hidden = np.zeros((1, hidden_size))
      self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.5
      self.bias_output = np.zeros((1, output_size))

.. dropdown:: Hint 2: Forward Pass
   :class-title: sd-font-weight-bold

   The forward pass computes output layer by layer:

   .. code-block:: python

      # Input to hidden
      self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
      self.hidden_output = sigmoid(self.hidden_input)

      # Hidden to output
      self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
      self.output = sigmoid(self.output_input)

.. dropdown:: Hint 3: Backward Pass
   :class-title: sd-font-weight-bold

   Backpropagation computes gradients layer by layer, from output to input:

   .. code-block:: python

      # Output layer gradient
      output_error = self.output - y
      output_delta = output_error * sigmoid_derivative(self.output_input)

      # Hidden layer gradient (error flows backward through weights)
      hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
      hidden_delta = hidden_error * sigmoid_derivative(self.hidden_input)

      # Update weights (gradient descent)
      self.weights_hidden_output -= self.learning_rate * np.dot(self.hidden_output.T, output_delta) / n_samples
      self.bias_output -= self.learning_rate * np.mean(output_delta, axis=0, keepdims=True)
      self.weights_input_hidden -= self.learning_rate * np.dot(X.T, hidden_delta) / n_samples
      self.bias_hidden -= self.learning_rate * np.mean(hidden_delta, axis=0, keepdims=True)

.. dropdown:: Complete Solution
   :class-title: sd-font-weight-bold

   .. code-block:: python
      :linenos:

      import numpy as np

      def sigmoid(x):
          return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

      def sigmoid_derivative(x):
          s = sigmoid(x)
          return s * (1 - s)

      class FeedforwardNetwork:
          def __init__(self, input_size=2, hidden_size=4, output_size=1, learning_rate=0.5):
              self.learning_rate = learning_rate
              self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.5
              self.bias_hidden = np.zeros((1, hidden_size))
              self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.5
              self.bias_output = np.zeros((1, output_size))

          def forward(self, X):
              self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
              self.hidden_output = sigmoid(self.hidden_input)
              self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
              self.output = sigmoid(self.output_input)
              return self.output

          def backward(self, X, y):
              n_samples = X.shape[0]
              output_error = self.output - y
              output_delta = output_error * sigmoid_derivative(self.output_input)
              hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
              hidden_delta = hidden_error * sigmoid_derivative(self.hidden_input)
              self.weights_hidden_output -= self.learning_rate * np.dot(self.hidden_output.T, output_delta) / n_samples
              self.bias_output -= self.learning_rate * np.mean(output_delta, axis=0, keepdims=True)
              self.weights_input_hidden -= self.learning_rate * np.dot(X.T, hidden_delta) / n_samples
              self.bias_hidden -= self.learning_rate * np.mean(hidden_delta, axis=0, keepdims=True)

          def train(self, X, y, epochs=5000):
              for epoch in range(epochs):
                  self.forward(X)
                  self.backward(X, y)

      # Test
      X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
      y = np.array([[0], [1], [1], [0]])
      np.random.seed(42)
      network = FeedforwardNetwork()
      network.train(X, y, epochs=5000)
      print("Accuracy:", np.mean(np.round(network.forward(X)) == y) * 100)

**Challenge Extension**: Add a second hidden layer to create a 2-4-4-1 architecture. You will need to add new weight matrices and modify both the forward and backward passes.

.. dropdown:: Challenge Solution
   :class-title: sd-font-weight-bold

   .. code-block:: python

      class TwoLayerNetwork:
          def __init__(self, learning_rate=0.5):
              self.learning_rate = learning_rate
              # Input to hidden1
              self.w1 = np.random.randn(2, 4) * 0.5
              self.b1 = np.zeros((1, 4))
              # Hidden1 to hidden2
              self.w2 = np.random.randn(4, 4) * 0.5
              self.b2 = np.zeros((1, 4))
              # Hidden2 to output
              self.w3 = np.random.randn(4, 1) * 0.5
              self.b3 = np.zeros((1, 1))

          def forward(self, X):
              self.z1 = np.dot(X, self.w1) + self.b1
              self.a1 = sigmoid(self.z1)
              self.z2 = np.dot(self.a1, self.w2) + self.b2
              self.a2 = sigmoid(self.z2)
              self.z3 = np.dot(self.a2, self.w3) + self.b3
              self.a3 = sigmoid(self.z3)
              return self.a3

          def backward(self, X, y):
              n = X.shape[0]
              # Layer 3
              d3 = (self.a3 - y) * sigmoid_derivative(self.z3)
              # Layer 2
              d2 = np.dot(d3, self.w3.T) * sigmoid_derivative(self.z2)
              # Layer 1
              d1 = np.dot(d2, self.w2.T) * sigmoid_derivative(self.z1)
              # Updates
              self.w3 -= self.learning_rate * np.dot(self.a2.T, d3) / n
              self.b3 -= self.learning_rate * np.mean(d3, axis=0, keepdims=True)
              self.w2 -= self.learning_rate * np.dot(self.a1.T, d2) / n
              self.b2 -= self.learning_rate * np.mean(d2, axis=0, keepdims=True)
              self.w1 -= self.learning_rate * np.dot(X.T, d1) / n
              self.b1 -= self.learning_rate * np.mean(d1, axis=0, keepdims=True)


Summary
=======

Key Takeaways
-------------

* **Single perceptrons** can only create linear decision boundaries and cannot solve XOR
* **Feedforward networks** with hidden layers can learn non-linear decision boundaries
* The **forward pass** computes output by transforming input through successive layers
* **Backpropagation** propagates error gradients backward to update all weights
* Each hidden neuron creates a linear boundary; combined, they form non-linear regions
* The **Universal Approximation Theorem** guarantees that one hidden layer is theoretically sufficient

Common Pitfalls
---------------

* **Too few hidden neurons**: Network may lack capacity to learn the pattern
* **Learning rate too high**: Causes oscillation or divergence
* **Learning rate too low**: Training takes excessively long
* **Poor weight initialization**: Symmetry can prevent learning; use small random values
* **Vanishing gradients**: Deep networks with sigmoid may have very small gradients
* **Not enough epochs**: Networks need many iterations for gradient descent to converge



References
==========

.. [Minsky1969] Minsky, M., & Papert, S. (1969). *Perceptrons: An Introduction to Computational Geometry*. MIT Press. ISBN: 978-0-262-63022-1 [Classic proof of perceptron limitations including XOR]

.. [Rumelhart1986] Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533-536. https://doi.org/10.1038/323533a0 [Popularized backpropagation for multi-layer networks]

.. [Hornik1989] Hornik, K., Stinchcombe, M., & White, H. (1989). Multilayer feedforward networks are universal approximators. *Neural Networks*, 2(5), 359-366. https://doi.org/10.1016/0893-6080(89)90020-8 [Proves universal approximation capability]

.. [LeCun2015] LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436-444. https://doi.org/10.1038/nature14539 [Comprehensive overview of deep learning]

.. [Goodfellow2016] Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. ISBN: 978-0-262-03561-3. https://www.deeplearningbook.org/ [Standard deep learning textbook]

.. [Haykin2009] Haykin, S. (2009). *Neural Networks and Learning Machines* (3rd ed.). Pearson. ISBN: 978-0-13-147139-9 [Comprehensive neural network reference]

.. [Nielsen2015] Nielsen, M. A. (2015). *Neural Networks and Deep Learning*. Determination Press. http://neuralnetworksanddeeplearning.com/ [Free online book with excellent explanations]

.. [Rosenblatt1958] Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain. *Psychological Review*, 65(6), 386-408. https://doi.org/10.1037/h0042519 [Original perceptron paper]

.. [NumPyDocs921] NumPy Developers. (2024). NumPy linear algebra (numpy.dot). *NumPy Documentation*. https://numpy.org/doc/stable/reference/generated/numpy.dot.html

.. [PillowDocs921] Clark, A., et al. (2024). *Pillow: Python Imaging Library* (Version 10.2.0). Python Software Foundation. https://pillow.readthedocs.io/
