"""Connect the individual NumPy layers into a small image-classification CNN."""

from FCL import Dense
from activation import ReLU
from convolution import Conv2D
from output import SoftmaxCrossEntropy
from pooling import MaxPool2D
from reshaping import Flatten


class CustomCNN:
    """A two-class network for one 3-channel, 4-by-4 image at a time."""

    def __init__(self):
        self.conv1 = Conv2D(num_filters=2, filter_size=3, stride=1, padding=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(pool_size=2, stride=2)
        self.flatten = Flatten()
        # For a 4x4 image, pooling leaves two 2x2 feature maps: 2 * 2 * 2 values.
        self.dense1 = Dense(input_dim=2 * 2 * 2, output_dim=2)
        self.loss_layer = SoftmaxCrossEntropy()

    def forward(self, image, label):
        """Return the two class scores and loss for a single 3x4x4 image."""
        if image.shape != (3, 4, 4):
            raise ValueError("CustomCNN expects one image shaped (3, 4, 4).")

        features = self.conv1.forward(image)
        features = self.relu1.forward(features)
        features = self.pool1.forward(features)
        features = self.flatten.forward(features)
        logits = self.dense1.forward(features)
        loss = self.loss_layer.forward(logits, label)
        return logits, loss

    def backward(self, lr=0.01):
        """Propagate the loss gradient through the network and update trainable layers."""
        gradient = self.loss_layer.backward()
        gradient = self.dense1.backward(gradient, lr)
        gradient = self.flatten.backward(gradient)
        gradient = self.pool1.backward(gradient)
        gradient = self.relu1.backward(gradient)
        self.conv1.backward(gradient, lr)
