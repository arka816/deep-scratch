# deep-scratch
  This repo is part of an attempt to develop various neural network models from scratch in python and providing alternative implementations of them for devices with CUDA-enabled GPUs.
## dependencies
  - Numpy 1.20.3
  - PyCUDA 2021.1
### installation
    pip install numpy pycuda
## description
### activations.py
  common activation functions e.g. sigmoid, tanh, ReLU, softmax and their derivatives
### layer.py
  implements a layer in an artificial neural network. supports common weight initialization schemes like glorot normal and glorot uniform.
### loss.py
  implements common loss functions (e.g. binary cross-entropy, mse, categorical crossentropy etc.) and their derivatives
### network.py
  contains the implementation of the ANN model with support for vector output. 
  batching like gradient descent, stochastic gradient descent, mini-batch gradient descent have been implemented.
  2nd moment based optimizers like adagrad, and RMSprop as well as ADAM based on both 1st and 2nd moments have been implemented.

## documentation
    model = network.Network(inputDim=inputDim, initializationScheme=initializationScheme)
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;inputDim: (*int*) the number of features in input (also the number of neurons in input layer) <br/>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;initializationScheme: (*string*) default: *randn*. can be any of the values below: <br />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1. randn: random normal with mean 0 <br />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2. he   : he initialization <br />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3. glorot-normal <br />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4. glorot-uniform <br />
    
    model.addLayer(dim=dim, activation=activation)
    
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;dim: (*int*) the number of hidden layers <br/>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;activation: (*string*) the activation function of the layer. can be any one of the following: <br/>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1. relu <br />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2. tanh <br />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3. sigmoid <br />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4. leaky-relu <br />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5. softmax <br />
  
    model.compile(loss=loss_func, optimizer=optimizer, batch_type=batch_type, batch_size=batch_size)
    
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;loss: (*string*) the loss function of the network. default: *binary-crossentropy*. can be any one of the following: <br/>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1. binary-crossentropy <br />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2. mse <br />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3. cat-crossentropy <br />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;optimizer: (*string*) the gradient descent optimizer for reducing loss. default: *gd*. can be any one of the following: <br/>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1. gd: gradient descent <br />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2. adagrad <br />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3. rmsprop <br />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4. adam <br />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;batch_type: (*string*) the batching for gradient descent. default: *bgd*. can be any one of the following: <br/>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1. bgd: batch gradient descent <br />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2. sgd: stochastic gradient descent <br />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3. mbgd: mini-batch gradient descent <br />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;batch_size: (*int*) batch size if mbgd is used for batching <br/>
  
    model.train(X_train, y_train, epochs=epochs, alpha=alpha, verbose=verbose)
    
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;epochs: (*int*) number of epochs to train the neural network. default: *100*
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;alpha: (*int*) learning rate for gradient descent. default: *0.1*
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;verbose: (*boolean*) default: *false*
  
    pred, accuracy = model.predict(X, y)
    
   returns
   1. pred: *numpy array* the prediction matrix
   2. accuracy: *float* the accuracy of prediction (if target variable is categorical)

  overall code to train and test:
  
    model = network.Network(inputDim=4, intializationScheme='glorot-uniform')
    model.addLayer(dim = 3, activation='sigmoid')
    model.addLayer(dim = 1, activation='sigmoid')
    model.compile(optimizer='adam', batch_type="mbgd", batch_size=32)
    model.train(X_train.T, y_train.T, alpha=0.1, epochs = 100)
    
## results

using a 2 layer neural ann with 300 hidden units and 784 (28 x 28 images) input units
  - achieved 97.78% accuracy over test set on training over grayscale images of handwritten letters
  - achieved around 96% accuracy over mnist digits dataset
