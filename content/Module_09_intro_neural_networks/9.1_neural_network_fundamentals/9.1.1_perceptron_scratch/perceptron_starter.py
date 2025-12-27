import numpy as np


class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        # TODO: Initialize weights with small random values
        self.weights = None  # Replace

        # TODO: Initialize bias to zero
        self.bias = None  # Replace

        # TODO: Store the learning rate
        self.learning_rate = None  # Replace

    def forward(self, x):
        # TODO: Compute weighted sum (dot product + bias)
        weighted_sum = None  # Replace

        # TODO: Apply step function (return 1 if >= 0, else 0)
        pass  # Replace

    def train(self, X, y, epochs=100):
        for epoch in range(epochs):
            errors = 0
            for i in range(len(X)):
                # TODO: Get prediction
                y_pred = None  # Replace

                # TODO: Calculate error
                error = None  # Replace

                # TODO: Update weights and bias if error != 0
                if error != 0:
                    pass  # Replace
                    errors += 1
            if errors == 0:
                print(f"Converged in {epoch + 1} epochs")
                return
        print(f"Training completed after {epochs} epochs")
