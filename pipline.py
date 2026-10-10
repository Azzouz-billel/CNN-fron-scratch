
from FCL import Dense
from activation import ReLU
from convolution import Conv2D
from general import SoftmaxCrossEntropy
from pooling import MaxPool2D
from reshaping import Flatten


class CustomCNN:
    """Modular CNN container chaining all layers together."""
    
    def __init__(self):
        self.conv1 = Conv2D(num_filters=2, filter_size=3, stride=1, padding=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(pool_size=2, stride=2)
        self.flatten = Flatten()
        self.dense1 = Dense(input_dim=2 * 2 * 2, output_dim=2)  # 2 classes
        self.loss_layer = SoftmaxCrossEntropy()

    def forward(self, image, label):
        x = self.conv1.forward(image)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)
        x = self.flatten.forward(x)
        logits = self.dense1.forward(x)
        loss = self.loss_layer.forward(logits, label)
        return logits, loss

    def backward(self, lr=0.01):
        grad = self.loss_layer.backward()
        grad = self.dense1.backward(grad, lr)
        grad = self.flatten.backward(grad)
        grad = self.pool1.backward(grad)
        grad = self.relu1.backward(grad)
        self.conv1.backward(grad, lr)

