import numpy as np

class SoftmaxCrossEntropy:
    """Computes Softmax probabilities and Categorical Cross-Entropy Loss."""
    
    def forward(self, logits, y_true):
        """
        logits: (num_classes, 1)
        y_true: scalar integer index or one-hot vector
        """
        # Shift logits for numerical stability
        exps = np.exp(logits - np.max(logits))
        self.probs = exps / np.sum(exps)
        self.y_true = y_true

        # Loss formula: -log(P_correct)
        loss = -np.log(self.probs[y_true] + 1e-15)
        return loss[0]

    def backward(self):
        """
        Gradient of Softmax + CrossEntropy Loss simplifies directly to: (Probs - OneHot)
        """
        d_out = self.probs.copy()
        d_out[self.y_true] -= 1.0
        return d_out

