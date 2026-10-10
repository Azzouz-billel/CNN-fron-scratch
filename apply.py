"""Run one demonstration training step through the hand-built CNN."""

import numpy as np

from pipline import CustomCNN


def main():
    # Fix the random seed so the example produces repeatable initial parameters/data.
    np.random.seed(42)
    sample_image = np.random.randn(3, 4, 4)
    target_class = 0

    model = CustomCNN()

    # A forward pass computes class scores and the loss for this labelled image.
    logits, loss = model.forward(sample_image, target_class)
    print(f"Initial loss     : {loss:.4f}")
    print(f"Predicted logits :\n{logits.flatten()}")

    # backward() propagates the loss gradient and updates the trainable parameters.
    model.backward(lr=0.05)

    # Re-evaluate the same example to show the effect of one parameter update.
    _, loss_after = model.forward(sample_image, target_class)
    print(f"Loss after 1 step: {loss_after:.4f}")


if __name__ == "__main__":
    main()
