"""A small NumPy convolution layer for single images in (channels, height, width) order."""

import numpy as np


class Conv2D:
    """Apply learnable square filters and propagate gradients through the operation."""

    def __init__(self, num_filters, filter_size, stride=1, padding=0):
        if num_filters <= 0 or filter_size <= 0 or stride <= 0 or padding < 0:
            raise ValueError("Filters and stride must be positive; padding cannot be negative.")

        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.W = None  # (filters, input channels, filter height, filter width)
        self.b = None  # One bias per output filter.

    def forward(self, X):
        """Return feature maps for an input shaped (channels, height, width)."""
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError("Conv2D expects input shaped (channels, height, width).")

        channels, height, width = X.shape
        if channels == 0:
            raise ValueError("Conv2D input must contain at least one channel.")
        filter_size, stride, padding = (
            self.filter_size,
            self.stride,
            self.padding,
        )

        if height + 2 * padding < filter_size or width + 2 * padding < filter_size:
            raise ValueError("The filter must fit inside the padded input.")

        if self.W is None:
            # He initialization keeps activations at a useful scale before ReLU.
            scale = np.sqrt(2.0 / (channels * filter_size * filter_size))
            self.W = np.random.randn(
                self.num_filters, channels, filter_size, filter_size
            ) * scale
            self.b = np.zeros(self.num_filters)
        elif self.W.shape[1] != channels:
            raise ValueError(
                f"This layer was initialized for {self.W.shape[1]} input channels, "
                f"but received {channels}."
            )

        if padding:
            self.X_padded = np.pad(
                X,
                ((0, 0), (padding, padding), (padding, padding)),
                mode="constant",
            )
        else:
            self.X_padded = X

        padded_height, padded_width = self.X_padded.shape[1:]
        out_height = (padded_height - filter_size) // stride + 1
        out_width = (padded_width - filter_size) // stride + 1
        out = np.zeros((self.num_filters, out_height, out_width))

        # Each output value is the sum of elementwise input/filter products plus bias.
        for filter_index in range(self.num_filters):
            for row in range(out_height):
                for column in range(out_width):
                    row_start, column_start = row * stride, column * stride
                    patch = self.X_padded[
                        :,
                        row_start : row_start + filter_size,
                        column_start : column_start + filter_size,
                    ]
                    out[filter_index, row, column] = (
                        np.sum(patch * self.W[filter_index]) + self.b[filter_index]
                    )

        self.last_input_shape = X.shape
        self.last_output_shape = out.shape
        return out

    def backward(self, d_out, lr=0.01):
        """Return the input gradient and update filters and biases using SGD."""
        if not hasattr(self, "last_input_shape"):
            raise RuntimeError("Call forward() before backward().")
        if d_out.shape != self.last_output_shape:
            raise ValueError(
                f"Expected output gradient shape {self.last_output_shape}, "
                f"got {d_out.shape}."
            )

        filter_size, stride, padding = (
            self.filter_size,
            self.stride,
            self.padding,
        )
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        # Gradients can be fractional even when the original image had integer pixels.
        gradient_dtype = np.result_type(
            self.X_padded.dtype, d_out.dtype, self.W.dtype
        )
        dX_padded = np.zeros(self.X_padded.shape, dtype=gradient_dtype)

        for filter_index in range(self.num_filters):
            for row in range(d_out.shape[1]):
                for column in range(d_out.shape[2]):
                    row_start, column_start = row * stride, column * stride
                    patch = self.X_padded[
                        :,
                        row_start : row_start + filter_size,
                        column_start : column_start + filter_size,
                    ]
                    gradient = d_out[filter_index, row, column]

                    # Accumulate contributions because filter windows can overlap.
                    dW[filter_index] += gradient * patch
                    db[filter_index] += gradient
                    dX_padded[
                        :, row_start : row_start + filter_size,
                        column_start : column_start + filter_size,
                    ] += gradient * self.W[filter_index]

        # Use the current filters for dX above, then update parameters for the next pass.
        self.W -= lr * dW
        self.b -= lr * db

        if padding:
            return dX_padded[:, padding:-padding, padding:-padding]
        return dX_padded
