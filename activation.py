"""Activation functions used between trainable CNN layers."""

import numpy as np


class ReLU:
    """Rectified Linear Unit: keep positive values and replace negatives with zero."""

    def forward(self, X):
        # Cache the input because its sign determines the derivative during backprop.
        self.last_input = X
        return np.maximum(0, X)

    def backward(self, d_out):
        """Pass gradients only through input elements that were strictly positive."""
        if not hasattr(self, "last_input"):
            raise RuntimeError("Call forward() before backward().")
        if d_out.shape != self.last_input.shape:
            raise ValueError("The gradient shape must match the input to forward().")

        d_in = d_out.copy()
        d_in[self.last_input <= 0] = 0
        return d_in
