"""Max-pooling layer for (channels, height, width) feature maps."""

import numpy as np


class MaxPool2D:
    """Downsample each channel by taking the maximum value in each window."""

    def __init__(self, pool_size=2, stride=2):
        if pool_size <= 0 or stride <= 0:
            raise ValueError("pool_size and stride must be positive.")
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, X):
        """Return pooled feature maps; incomplete edge windows are ignored."""
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError("MaxPool2D expects input shaped (channels, height, width).")

        channels, height, width = X.shape
        if channels == 0:
            raise ValueError("MaxPool2D input must contain at least one channel.")
        pool_size, stride = self.pool_size, self.stride
        if height < pool_size or width < pool_size:
            raise ValueError("The pooling window must fit inside the input.")

        out_height = (height - pool_size) // stride + 1
        out_width = (width - pool_size) // stride + 1
        out = np.zeros((channels, out_height, out_width), dtype=X.dtype)
        self.last_input = X

        for channel in range(channels):
            for row in range(out_height):
                for column in range(out_width):
                    row_start, column_start = row * stride, column * stride
                    patch = X[
                        channel,
                        row_start : row_start + pool_size,
                        column_start : column_start + pool_size,
                    ]
                    out[channel, row, column] = np.max(patch)

        self.last_output_shape = out.shape
        return out

    def backward(self, d_out):
        """Route each pooled gradient to the first maximum in its source window."""
        if not hasattr(self, "last_input"):
            raise RuntimeError("Call forward() before backward().")
        if d_out.shape != self.last_output_shape:
            raise ValueError(
                f"Expected output gradient shape {self.last_output_shape}, "
                f"got {d_out.shape}."
            )

        X = self.last_input
        channels = X.shape[0]
        pool_size, stride = self.pool_size, self.stride
        # Keep fractional derivatives when the original feature values were integers.
        dX = np.zeros(
            X.shape, dtype=np.result_type(X.dtype, d_out.dtype)
        )

        for channel in range(channels):
            for row in range(d_out.shape[1]):
                for column in range(d_out.shape[2]):
                    row_start, column_start = row * stride, column * stride
                    patch = X[
                        channel,
                        row_start : row_start + pool_size,
                        column_start : column_start + pool_size,
                    ]

                    # argmax selects one location, preventing duplicate gradients on ties.
                    max_row, max_column = np.unravel_index(
                        np.argmax(patch), patch.shape
                    )
                    dX[channel, row_start + max_row, column_start + max_column] += (
                        d_out[channel, row, column]
                    )

        return dX
