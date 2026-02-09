"""
Exercise 9.1.1 - Perceptron (Starter Template)

Fill in the TODO sections to build a working perceptron classifier.

Inspired by:
  Rosenblatt, F. (1958). The Perceptron: A Probabilistic Model for
  Information Storage and Organization in the Brain.
  Psychological Review, 65(6), 386-408.

  scikit-learn Perceptron
  https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html
"""

import numpy as np


class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        # TODO: Initialize weights with small random values
        # (Small values prevent large initial outputs; randomness
        #  gives each input a different starting influence)
        self.weights = None  # Replace

        # TODO: Initialize bias to zero
        # (The bias shifts the decision boundary; zero is a neutral start)
        self.bias = None  # Replace

        # TODO: Store the learning rate
        # (It controls how big each weight update is)
        self.learning_rate = None  # Replace

    def forward(self, x):
        # TODO: Compute weighted sum (dot product of weights and x, plus bias)
        # Formula:  z = w1*x1 + w2*x2 + ... + wn*xn + b
        weighted_sum = None  # Replace

        # TODO: Apply step function (return 1 if z >= 0, else 0)
        pass  # Replace

    def train(self, X, y, epochs=100):
        for epoch in range(epochs):
            errors = 0
            for i in range(len(X)):
                # TODO: Get prediction using forward()
                y_pred = None  # Replace

                # TODO: Calculate error (true label minus prediction)
                error = None  # Replace

                # TODO: If error is not zero, update weights and bias
                # using the perceptron learning rule:
                #   weights += learning_rate * error * input
                #   bias    += learning_rate * error
                if error != 0:
                    pass  # Replace with weight and bias updates
                    errors += 1
            if errors == 0:
                print(f"Converged in {epoch + 1} epochs")
                return
        print(f"Training completed after {epochs} epochs")
