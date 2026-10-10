import numpy as np


class Dense:
    """Standard fully connected linear layer: Y = W @ X + b"""
    
    def __init__(self, input_dim, output_dim):
        # Xavier weight initialization
        self.W = np.random.randn(output_dim, input_dim) * np.sqrt(2.0 / input_dim)
        self.b = np.zeros((output_dim, 1))

    def forward(self, X):
        self.last_input = X  # Shape: (input_dim, 1)
        return np.dot(self.W, X) + self.b

    def backward(self, d_out, lr=0.01):
        """
        d_out shape: (output_dim, 1)
        Returns dX shape: (input_dim, 1)
        """
        dW = np.dot(d_out, self.last_input.T)
        db = d_out
        dX = np.dot(self.W.T, d_out)

        # Update weights and bias
        self.W -= lr * dW
        self.b -= lr * db

        return dX
