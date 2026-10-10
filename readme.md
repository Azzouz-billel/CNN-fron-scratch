# CNN From Scratch

This project is a hands-on study of how Convolutional Neural Networks work internally.

The goal is to understand the real mechanics behind CNNs, not just use a high-level framework. We focus on the core operations that make CNNs powerful for images and computer vision.

## What we are learning

We will break down and implement the main building blocks of a CNN:

- convolution
- kernels and filters
- feature maps
- activation functions such as ReLU
- max pooling
- flattening
- fully connected layers
- softmax and cross-entropy loss
- backpropagation and parameter updates

## Why this matters

CNNs are designed to detect patterns in images such as edges, textures, corners, and shapes. Instead of treating images as flat vectors, they preserve spatial relationships and learn local structures efficiently.

This project helps us understand:

- how a filter scans an image
- how values are summed and combined
- how activation functions introduce non-linearity
- why pooling reduces dimensions
- how a network learns by adjusting weights

## Project structure

The folder contains a simple NumPy-based CNN implementation with separate files for each concept:

- `convolution.py` — 2D convolution layer
- `activation.py` — ReLU activation
- `pooling.py` — max pooling
- `reshaping.py` — flattening
- `FCL.py` — fully connected layer
- `output.py` — softmax + cross-entropy loss
- `pipline.py` — CNN pipeline combining the layers
- `apply.py` — example forward/backward training pass

## How it works

The pipeline follows a basic CNN flow:

1. Input image enters the convolution layer
2. Filters extract features
3. ReLU applies non-linearity
4. Pooling reduces spatial size
5. Features are flattened
6. Dense layer produces class scores
7. Loss is computed and gradients are propagated backward

## Run the example

From this folder, run:

```bash
python apply.py
```

This script creates a random sample image, performs a forward pass, computes the loss, updates the network through backpropagation, and prints the loss before and after one training step.

## Learning approach

This project is meant to be educational and transparent.

Instead of hiding the math behind frameworks, we look directly at:

- the shapes of tensors
- the summation inside convolution
- the role of the activation function
- how gradients flow through the network

## Goal

The final goal is to understand how CNNs really work under the hood and how each layer contributes to image understanding and learning.

This is not just a model implementation — it is a practical way to explore the fundamentals of CNNs step by step.
