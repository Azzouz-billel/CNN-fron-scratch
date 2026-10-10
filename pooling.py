import numpy as np


class MaxPool2D:
    """Max Pooling Layer for spatial downsampling."""
    
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, X):
        """
        Input X shape: (C, H, W)
        Output shape:  (C, H / pool_size, W / pool_size)
        """
        self.last_input = X
        C, H, W = X.shape
        f, s = self.pool_size, self.stride

        H_out = (H - f) // s + 1
        W_out = (W - f) // s + 1

        out = np.zeros((C, H_out, W_out))

        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h_st, w_st = i * s, j * s
                    patch = X[c, h_st:h_st + f, w_st:w_st + f]
                    out[c, i, j] = np.max(patch)

        return out

    def backward(self, d_out):
        """Routes gradient only to the index of maximum value during forward pass."""
        X = self.last_input
        C, H, W = X.shape
        f, s = self.pool_size, self.stride
        _, H_out, W_out = d_out.shape

        dX = np.zeros_like(X)

        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h_st, w_st = i * s, j * s
                    patch = X[c, h_st:h_st + f, w_st:w_st + f]
                    max_val = np.max(patch)
                    
                    # Create binary mask for max value position
                    mask = (patch == max_val)
                    dX[c, h_st:h_st + f, w_st:w_st + f] += mask * d_out[c, i, j]

        return dX