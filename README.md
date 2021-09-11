# deep-scratch
  This repo is part of an attempt to develop various neural network models from scratch in python and providing alternative implementations of them for devices with CUDA-enabled GPUs.
## dependencies
  - Numpy 1.20.3
  - PyCUDA 2021.1
## installation
    pip install numpy pycuda
## description
### activations.py
  common activation functions e.g. sigmoid, tanh, ReLU and their derivatives
### layer.py
  implements a layer in an artificial neural network
### loss.py
  implements common loss functions (e.g. binary cross-entropy, mse etc.) and their derivatives
### network.py
  contains the implementation of the ANN model. For now this only supports one neuron in output layer (unidimensional output). 
  optimizers like gradient descent, stochastic gradient descent, mini-batch gradient descent have been implemented.
