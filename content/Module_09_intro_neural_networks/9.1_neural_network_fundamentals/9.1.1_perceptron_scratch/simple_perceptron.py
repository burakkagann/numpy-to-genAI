import numpy as np

np.random.seed(42)


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


# Generate linearly separable data
X = np.vstack([
    np.random.randn(50, 2) * 0.5 + [-1, -1],
    np.random.randn(50, 2) * 0.5 + [1, 1]
])
y = np.array([0] * 50 + [1] * 50)

# Train perceptron
print("Training perceptron...")
perceptron = Perceptron(input_size=2, learning_rate=0.1)
perceptron.train(X, y)
print(f"Final weights: {perceptron.weights}")
print(f"Final bias: {perceptron.bias}")
