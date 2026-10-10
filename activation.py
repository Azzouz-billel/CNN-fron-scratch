import numpy as np
# relu
class ReLU:
    """Rectified Linear Unit activation layer: max(0, x)."""
    
    def forward(self, X):
        self.last_input = X
        return np.maximum(0, X)

    def backward(self, d_out):
        """Pass gradient only where input was greater than 0."""
        d_in = d_out.copy()
        d_in[self.last_input <= 0] = 0
        return d_in
