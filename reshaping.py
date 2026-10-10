"""Shape conversion layer used between convolutional features and a dense layer."""

from math import prod


class Flatten:
    """Turn a feature tensor into a column vector without changing its element order."""

    def forward(self, X):
        # Save the exact incoming shape so backward can restore all feature dimensions.
        self.orig_shape = X.shape
        return X.reshape(-1, 1)

    def backward(self, d_out):
        """Restore the gradient to the shape of the tensor that entered forward()."""
        if not hasattr(self, "orig_shape"):
            raise RuntimeError("Call forward() before backward().")
        expected_size = prod(self.orig_shape)
        if d_out.size != expected_size:
            raise ValueError(
                f"Gradient has {d_out.size} values; expected {expected_size}."
            )
        return d_out.reshape(self.orig_shape)
