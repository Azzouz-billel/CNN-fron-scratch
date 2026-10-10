import numpy as np

class Conv2D:
    """2D Convolutional Layer with learnable filters and biases."""
    
    def __init__(self, num_filters, filter_size, stride=1, padding=0):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.W = None  # Shape: (num_filters, C_in, f, f)
        self.b = None  # Shape: (num_filters, 1)

    def forward(self, X):
        """
        Input X shape: (C_in, H, W)
        Output shape:  (num_filters, H_out, W_out)
        """
        C_in, H, W = X.shape
        f, s, p = self.filter_size, self.stride, self.padding

        # He / He-normal parameter initialization
        if self.W is None:
            self.W = np.random.randn(self.num_filters, C_in, f, f) * np.sqrt(2.0 / (C_in * f * f))
            self.b = np.zeros((self.num_filters, 1))

        # Pad spatial dimensions
        if p > 0:
            self.X_padded = np.pad(X, ((0, 0), (p, p), (p, p)), mode='constant')
        else:
            self.X_padded = X

        H_pad, W_pad = self.X_padded.shape[1], self.X_padded.shape[2]
        H_out = (H_pad - f) // s + 1
        W_out = (W_pad - f) // s + 1

        out = np.zeros((self.num_filters, H_out, W_out))

        # Forward spatial convolution
        for n in range(self.num_filters):
            for i in range(H_out):
                for j in range(W_out):
                    h_st, w_st = i * s, j * s
                    slice_3d = self.X_padded[:, h_st:h_st + f, w_st:w_st + f]
                    out[n, i, j] = np.sum(slice_3d * self.W[n]) + self.b[n]

        self.last_input_shape = X.shape
        return out

    def backward(self, d_out, lr=0.01):
        """
        Computes gradients with respect to weights (dW), biases (db),
        updates parameters, and returns input gradient (dX).
        """
        C_in, H, W = self.last_input_shape
        f, s, p = self.filter_size, self.stride, self.padding
        num_filters, H_out, W_out = d_out.shape

        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dX_padded = np.zeros_like(self.X_padded)

        for n in range(num_filters):
            for i in range(H_out):
                for j in range(W_out):
                    h_st, w_st = i * s, j * s
                    slice_3d = self.X_padded[:, h_st:h_st + f, w_st:w_st + f]

                    # Gradient wrt weights and biases
                    dW[n] += d_out[n, i, j] * slice_3d
                    db[n] += d_out[n, i, j]

                    # Gradient wrt padded input
                    dX_padded[:, h_st:h_st + f, w_st:w_st + f] += d_out[n, i, j] * self.W[n]

        # Parameter update step
        self.W -= lr * dW
        self.b -= lr * db

        # Crop out padding to return gradient matching original input shape
        if p > 0:
            return dX_padded[:, p:-p, p:-p]
        return dX_padded