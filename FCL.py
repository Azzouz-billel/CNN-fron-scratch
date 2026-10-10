"""A single-example fully connected layer implemented with NumPy arrays."""

import numpy as np


class Dense:
    """Linear layer computing ``W @ X + b`` for column-vector inputs."""

    def __init__(self, input_dim, output_dim):
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive.")

        # He initialization is appropriate here because this network uses ReLU.
        self.W = np.random.randn(output_dim, input_dim) * np.sqrt(2.0 / input_dim)
        self.b = np.zeros((output_dim, 1))

    def forward(self, X):
        """Map an input column vector of shape (input_dim, 1) to output scores."""
        if X.shape != (self.W.shape[1], 1):
            raise ValueError(
                f"Expected input shape {(self.W.shape[1], 1)}, got {X.shape}."
            )
        self.last_input = X
        return self.W @ X + self.b

    def backward(self, d_out, lr=0.01):
        """Return the input gradient, then update weights and bias with SGD."""
        if not hasattr(self, "last_input"):
            raise RuntimeError("Call forward() before backward().")
        if d_out.shape != (self.W.shape[0], 1):
            raise ValueError(
                f"Expected output gradient shape {(self.W.shape[0], 1)}, "
                f"got {d_out.shape}."
            )

        # Compute dX before the update so it uses the same weights as forward().
        dX = self.W.T @ d_out
        dW = d_out @ self.last_input.T
        self.W -= lr * dW
        self.b -= lr * d_out
        return dX
