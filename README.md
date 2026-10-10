# CNN From Scratch

A small NumPy project for learning how the main parts of a Convolutional Neural
Network (CNN) work. The layers are implemented directly so their calculations
and gradients are visible rather than hidden behind a deep-learning framework.

## What this project demonstrates

- 2D convolution with multiple input channels, filters, stride, and zero padding
- ReLU activation
- max pooling and its backward gradient
- flattening feature maps into a column vector
- a fully connected layer
- numerically stable softmax and categorical cross-entropy
- manual backpropagation and stochastic gradient descent updates

## CNN flow

For one input image, the example runs this sequence:

1. Convolution extracts local features.
2. ReLU keeps positive activations.
3. Max pooling downsamples each feature map.
4. Flatten converts the feature maps to a vector.
5. The dense layer produces two class scores (logits).
6. Softmax cross-entropy computes the loss.
7. Backpropagation sends gradients through the layers and updates parameters.

## Run the example

From this directory, install the dependency and run the one-step demonstration:

```bash
python -m pip install -r requirements.txt
python apply.py
```

`apply.py` generates a deterministic random image of shape `(3, 4, 4)`, trains
on class `0` for one update, and prints the logits and loss before and after the
update. It is a tiny learning demonstration, not a trained image classifier.

## Files

- `convolution.py` — convolution layer and its gradients
- `activation.py` — ReLU layer
- `pooling.py` — max-pooling layer
- `reshaping.py` — flatten and restore tensor shapes
- `FCL.py` — fully connected layer
- `output.py` — softmax and cross-entropy loss
- `pipline.py` — network that connects the layers (filename retained as-is)
- `apply.py` — runnable training-step example
- `requirements.txt` — NumPy dependency

## Scope and learning notes

This implementation processes one image at a time and uses channels-first
arrays shaped `(channels, height, width)`. The assembled `CustomCNN` is
intentionally fixed to three-channel `4 x 4` images and two output classes.
It is designed for understanding the operations and tensor shapes; it does not
include datasets, batches, model saving, or framework-level performance
optimizations.
