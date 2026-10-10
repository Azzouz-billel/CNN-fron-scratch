import numpy as np

class Flatten:
    """Converts multi-dimensional feature tensor into a 1D vector."""
    
    def forward(self, X):
        self.orig_shape = X.shape  # Store (C, H, W)
        return X.flatten().reshape(-1, 1)  # Output shape: (C*H*W, 1)

    def backward(self, d_out):
        """Reshapes 1D gradient vector back to multi-dimensional tensor shape."""
        return d_out.reshape(self.orig_shape)

