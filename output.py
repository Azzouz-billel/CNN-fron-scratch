"""Numerically stable softmax and categorical cross-entropy for one example."""

import numpy as np


class SoftmaxCrossEntropy:
    """Convert class scores to probabilities and compute their classification loss."""

    def forward(self, logits, y_true):
        """
        Compute loss for scores shaped (classes, 1).

        ``y_true`` may be a class index or a one-hot vector with one value per class.
        """
        logits = np.asarray(logits)
        if logits.ndim != 2 or logits.shape[1] != 1 or logits.shape[0] == 0:
            raise ValueError("logits must have shape (num_classes, 1).")

        class_count = logits.shape[0]
        if isinstance(y_true, (int, np.integer)) and not isinstance(
            y_true, (bool, np.bool_)
        ):
            if y_true < 0 or y_true >= class_count:
                raise ValueError("The class index is outside the logits range.")
            target = np.zeros_like(logits)
            target[y_true, 0] = 1
        else:
            target = np.asarray(y_true)
            if target.shape == (class_count,):
                target = target.reshape(class_count, 1)
            if target.shape != logits.shape:
                raise ValueError(
                    f"One-hot labels must have shape {(class_count,)} "
                    f"or {logits.shape}."
                )
            if not np.all((target == 0) | (target == 1)) or np.sum(target) != 1:
                raise ValueError("One-hot labels must contain exactly one 1.")

        # Subtracting the largest score prevents exponential overflow.
        shifted_logits = logits - np.max(logits)
        exponentials = np.exp(shifted_logits)
        self.probs = exponentials / np.sum(exponentials)
        self.target = target

        # Clip only inside log so an extremely small probability stays finite.
        return float(-np.sum(target * np.log(np.maximum(self.probs, 1e-15))))

    def backward(self):
        """Return the derivative of softmax plus cross-entropy: probabilities - target."""
        if not hasattr(self, "probs"):
            raise RuntimeError("Call forward() before backward().")
        return self.probs - self.target
