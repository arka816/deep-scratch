import numpy as np
import activations
import loss
import layer
import matplotlib.pyplot as plt
import json

class Network:
    def __init__(self, inputDim, intializationScheme = 'randn'):
        # layers : hidden layers + output layer (we assume 1 output neuron in the output layer)
        # dims   : number of neurons in each layer (input + hidden + output layer)
        # activations : activation of outputs of each layer stored for backpropagation (a)
        # outputs     : outputs of each layer stored for backpropagation (z)
        self.activations = []
        self.outputs = []
        self.layers = []
        self.dims = [inputDim]
        self.intializationScheme = intializationScheme

    def addLayer(self, dim, activation):
        if dim == 0:
            raise Exception("layer size cannot be zero")

        self.dims.append(dim)
        self.layers.append(layer.Layer(self.dims[-2], self.dims[-1], activation, self.intializationScheme))

    def compile(self, loss="binary-crossentropy", optimizer='gd', batch_type='bgd', batch_size = 0):
        # loss      : the loss metric used to backpropagate and adjust weights
        # optimizer : the loss optimization method (gd, adaptive learning, ADAM, RADAM etc.)
        # batch_type: the type of batching (bgd, sgd, mbgd etc)
        # batch_size: the batch size if mini-batch gradient descent is used as optimizer

        self.loss = loss
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.batch_type = batch_type

        if self.optimizer == 'adagrad' or self.optimizer == 'rmsprop':
            # is momentum based optimizer
            self.momentum_W = []
            self.momentum_b = []

            for i in range(len(self.dims) - 1):
                m_W = np.zeros((self.dims[i], self.dims[i + 1]))
                m_b = np.zeros((self.dims[i+1], 1))
                self.momentum_W.append(m_W)
                self.momentum_b.append(m_b)

        if self.optimizer == 'adam':
            # adam uses both first and second moments
            self.m_W = []
            self.m_b = []
            self.v_W = []
            self.v_b = []

            for i in range(len(self.dims) - 1):
                m_W = np.zeros((self.dims[i], self.dims[i + 1]))
                m_b = np.zeros((self.dims[i+1], 1))
                v_W = np.zeros((self.dims[i], self.dims[i + 1]))
                v_b = np.zeros((self.dims[i+1], 1))

                self.m_W.append(m_W)
                self.m_b.append(m_b)    
                self.v_W.append(v_W)
                self.v_b.append(v_b)            

        if batch_type == 'mbgd':
            if batch_size == 0:
                raise Exception("unspecified batch size for mini batch gradient descent")
            elif batch_size < 0:
                raise Exception("invalid batch size for mini-batch gradient descent")

    def optimize(self, grad_W, grad_b, alpha, epsilon=1e-4, beta=0.9, beta_1=0.9, beta_2=0.99):
        # grad_W : list of gradients of loss w.r.t. W's of all layers
        # grad_b : list of gradients of loss w.r.t. b's of all layers
        # alpha  : learning rate of gradient descent
        # epsilon: denominator weight for adagrad
        # beta   : weight for rms propagation
        # beta_1 : beta 1 for adam (weight for first moment)
        # beta_2 : beta 2 for adam (weight for second moment)

        self.gen += 1

        if self.optimizer == 'gd':
            # Classical Gradient Descent
            for layer in range(len(self.layers)):
                self.layers[layer].W -= alpha * grad_W[layer]
                self.layers[layer].b -= alpha * grad_b[layer]

        elif self.optimizer == 'adagrad':
            # adagrad algorithm
            # suitable for sparse features
            for layer in range(len(self.layers)):
                dW, db = grad_W[layer], grad_b[layer]
                self.momentum_W[layer] += dW ** 2
                self.momentum_b[layer] += db ** 2

                self.layers[layer].W -= (alpha / (epsilon + np.sqrt(self.momentum_W[layer]))) * grad_W[layer]
                self.layers[layer].b -= (alpha / (epsilon + np.sqrt(self.momentum_b[layer]))) * grad_b[layer]

        elif self.optimizer == 'rmsprop':
            # root mean square propagation
            # removes the possibility of learning rate dampening of adagrad for dense features
            for layer in range(len(self.layers)):
                dW, db = grad_W[layer], grad_b[layer]
                self.momentum_W[layer] = beta * self.momentum_W[layer] + (1 - beta) * (dW ** 2)
                self.momentum_b[layer] = beta * self.momentum_b[layer] + (1 - beta) * (db ** 2)

                self.layers[layer].W -= (alpha / (epsilon + np.sqrt(self.momentum_W[layer]))) * grad_W[layer]
                self.layers[layer].b -= (alpha / (epsilon + np.sqrt(self.momentum_b[layer]))) * grad_b[layer]

        elif self.optimizer == 'adam':
            # adaptive moment estimation
            for layer in range(len(self.layers)):
                dW, db = grad_W[layer], grad_b[layer]

                self.m_W[layer] = beta_1 * self.m_W[layer] + (1 - beta_1) * dW
                self.m_b[layer] = beta_1 * self.m_b[layer] + (1 - beta_1) * db

                self.v_W[layer] = beta_2 * self.v_W[layer] + (1 - beta_2) * (dW ** 2)
                self.v_b[layer] = beta_2 * self.v_b[layer] + (1 - beta_2) * (db ** 2)

                # calculate unbiased estimates
                m_W = self.m_W[layer] / (1 - beta_1 ** self.gen)
                m_b = self.m_b[layer] / (1 - beta_1 ** self.gen)

                v_W = self.v_W[layer] / (1 - beta_2 ** self.gen)
                v_b = self.v_b[layer] / (1 - beta_2 ** self.gen)

                self.layers[layer].W -= (alpha / (epsilon + np.sqrt(v_W))) * m_W
                self.layers[layer].b -= (alpha / (epsilon + np.sqrt(v_b))) * m_b


    def calculate_accuracy(self, y_pred, y_train):
        if self.loss == 'binary-crossentropy':
            y_pred = np.where(y_pred >= 0.5, 1, 0)
            return (y_pred == y_train).all(axis = 0).mean()
        elif self.loss == 'cat-crossentropy':
            y_pred = (y_pred == np.max(y_pred, axis=0)).astype(int)
            # print(y_pred[0:5])
            return (y_pred == y_train).all(axis = 0).mean()

    def calculate_loss(self, y_pred, y_train):
        if self.loss == 'binary-crossentropy':
            pred_loss = loss.binary_crossentropy(y_pred, y_train)
        elif self.loss == 'mse':
            pred_loss = loss.mse(y_pred, y_train)
        elif self.loss == 'cat-crossentropy':
            pred_loss = loss.cat_crossentropy(y_pred, y_train)
        else:
            raise Exception('loss metric not defined')
        return pred_loss

    def train(self, X_train, y_train, epochs = 100, alpha = 0.1, verbose = False):
        if self.dims[0] != X_train.shape[0]:
            raise Exception('Dimension of input data and input layer do not match')

        # initial activation value is output from input layer (input to the model)
        self.gen = 0
        accuracies = []
        losses = []

        if self.batch_type == 'bgd':
            # BATCH GRADIENT DESCENT

            self.activations.append(X_train)

            for i in range(epochs):
                # Forward Propagation
                y_pred = self.forward_pass(X_train)

                # Backward propagation
                grad_w, grad_b = self.backward_pass(y_train)

                # Weight Update
                self.optimize(grad_w, grad_b, alpha)

                # Calculate loss
                pred_loss = self.calculate_loss(y_pred, y_train)

                # calculate accuracy
                accuracy = self.calculate_accuracy(y_pred, y_train)
                
                accuracies.append(accuracy)
                losses.append(pred_loss)

                if verbose:
                    print(f'Epoch {i+1}: calculated loss = {pred_loss} | calculated accuracy = {accuracy}')
                
                self.activations = [X_train]
                self.outputs = []

        elif self.batch_type == 'sgd':
            # STOCHASTIC GRADIENT DESCENT

            for i in range(epochs):
                y_pred = []
                for j in range(X_train.shape[1]):
                    self.activations = [X_train.T[j:j+1].T]
                    self.outputs = []

                    # Forward Propagation
                    y = self.forward_pass(X_train.T[j:j+1].T)
                    y_pred.append(y[0])

                    # Backward propagation
                    grad_w, grad_b = self.backward_pass(y_train.T[j:j+1].T)

                    # Weight Update
                    self.optimize(grad_w, grad_b, alpha)
                
                y_pred = np.array(y_pred).T

                # Calculate loss
                pred_loss = self.calculate_loss(y_pred, y_train)

                # calculate accuracy
                accuracy = self.calculate_accuracy(y_pred, y_train)
                
                accuracies.append(accuracy)
                losses.append(pred_loss)

                if verbose:
                    print(f'Epoch {i+1}: calculated loss = {pred_loss} | calculated accuracy = {accuracy}')

        elif self.batch_type == 'mbgd':
            # MINI BATCH GRADIENT DESCENT

            sample_size = X_train.shape[1]

            for i in range(epochs):
                j = 0
                y_pred = []

                while j < sample_size:
                    end = min(sample_size, j + self.batch_size)

                    self.activations = [X_train.T[j: end].T]
                    self.outputs = []

                    # Forward Propagation
                    y_batch = self.forward_pass(X_train.T[j:end].T)
                    y_pred.append(y_batch)

                    # Backward propagation
                    grad_w, grad_b = self.backward_pass(y_train.T[j:end].T)

                    # Weight Update
                    self.optimize(grad_w, grad_b, alpha)

                    j += self.batch_size

                y_pred = np.concatenate(y_pred, axis = 1)

                # Calculate loss
                pred_loss = self.calculate_loss(y_pred, y_train)

                # Calculate accuracy
                accuracy = self.calculate_accuracy(y_pred, y_train)
                
                accuracies.append(accuracy)
                losses.append(pred_loss)

                if verbose:
                    print(f'Epoch {i+1}: calculated loss = {pred_loss:.12f} | calculated accuracy = {accuracy:.5f}')

        # plot the losses and accuracy development
        if verbose:
            xrange = np.arange(0, epochs)
            plt.plot(xrange, losses, label="loss")
            plt.title("loss over epochs")
            plt.show()
            plt.plot(xrange, accuracies, label="accuracy")
            plt.title("accuracy over epochs")
            plt.show()
        
    def predict(self, X_test, y_test):
        y_pred = self.forward_pass(X_test)

        # confusion matrix (gives an idea of the models classification 
        # power in terms of true positives and false negatives)
        # cf = np.array([
        #     [np.sum((y_pred == y_test) & (y_test == 1)), np.sum((y_pred != y_test) & (y_test == 1))],
        #     [np.sum((y_pred != y_test) & (y_test == 0)), np.sum((y_pred == y_test) & (y_test == 0))]
        # ])

        #accuracy score
        accuracy = self.calculate_accuracy(y_pred, y_test)

        return y_pred, accuracy

    def one_layer_forward_pass(self, W, b, a, activation):
        # simulates a forward propagation for the entire dataset through one layer
        # Input:
        # W : weight matrix of current layer (Nl-1 x Nl)
        # b : bias value of current layer 
        # a : activation values from previous layer (input value for input layer) (Nl-1 x m)
        # z : W.a + b (Nl x m)
        # Output : 
        # a : activation function operated on z (Nl x m)

        z = np.dot(W.T, a) + b

        if activation == 'relu':
            a = activations.ReLU(z)
        elif activation == 'tanh':
            a = activations.tanh(z)
        elif activation == 'sigmoid':
            a = activations.sigmoid(z)
        elif activation == 'leaky-relu':
            a = activations.leaky_ReLU(z)
        elif activation == 'softmax':
            a = activations.softmax(z)
        else:
            raise Exception("activation function not recognised")

        return a, z

    def forward_pass(self, X_train):
        # simulates one forward propagation (one epoch) for the entire dataset
        # input : training feature matrix (X)
        # output: simulated output (calculated y vector)

        num_layers = len(self.layers)

        a = X_train

        for i in range(num_layers):
            W = self.layers[i].W
            b = self.layers[i].b
            activation = self.layers[i].activation

            a, z = self.one_layer_forward_pass(W, b, a, activation)
            self.activations.append(a)
            self.outputs.append(z)

        return a

    def one_layer_backward_pass(self, activation, z_curr, da_curr, W_curr, a_prev, y_pred, y_train):
        # simulates a backpropagation for the entire dataset through 1 layer
        # Input:
        # activation : activation function
        # z_curr  : z of the current layer (Nl x m)
        # da_curr : deriavtive of loss function w.r.t. a of the current layer (Nl x m)
        # W_curr  : weight matrix from previous layer to this layer (Nl-1 x Nl)
        # a_prev  : output from the previous layer (Nl-1 x m)
        # Output:
        # da_prev : derivative of loss function w.r.t. a of the previous layer (Nl-1 x m)
        # dw_curr : derivative of loss function w.r.t. W_curr (Nl-1 x Nl)
        # db_curr : derivative of loss function w.r.t. bias term of current layer

        m = a_prev.shape[1]

        if activation == 'relu':
            d_activation = activations.d_relu
        elif activation == 'sigmoid':
            d_activation = activations.d_sigmoid
        elif activation == 'tanh':
            d_activation = activations.d_tanh
        elif activation == 'leaky-relu':
            d_activation = activations.d_leaky_relu
        elif activation == 'softmax':
            d_activation = activations.d_softmax
        else:
            raise Exception("activation function not recognised")

        if activation == 'softmax':
            dz_curr = y_pred - y_train
        else:
            dg_curr = d_activation(z_curr)
            dz_curr = da_curr * dg_curr
        da_prev = np.dot(W_curr, dz_curr)
        dw_curr = np.dot(a_prev, dz_curr.T) / m
        db_curr = np.sum(dz_curr, axis = 1, keepdims=True) / m

        return da_prev, dw_curr, db_curr

    def backward_pass(self, y_train):
        # simulates one epoch of backward propagation
        # input  : y_train (actual output)
        # output : gradient of loss function w.r.t. W's and b's of all layers

        num_layers = len(self.layers)
        a_curr = self.activations[-1]
        y_pred = a_curr
        y_train = y_train.reshape(y_pred.shape)

        if self.loss == 'binary-crossentropy':
            da_prev = loss.d_binary_crossentropy(y_pred, y_train)
        elif self.loss == 'mse':
            da_prev = loss.d_mse(y_pred, y_train)
        elif self.loss == 'cat-crossentropy':
            da_prev = loss.d_cat_crossentropy(y_pred, y_train)
        else:
            raise Exception("loss metric not defined")

        grad_W = []
        grad_b = []

        for i in range(num_layers-1, -1, -1):
            activation = self.layers[i].activation
            da_curr = da_prev
            W_curr = self.layers[i].W
            z_curr = self.outputs[i]
            a_prev = self.activations[i]

            da_prev, dw_curr, db_curr = self.one_layer_backward_pass(activation, z_curr, da_curr, W_curr, a_prev, y_pred, y_train)
    
            grad_W.insert(0, dw_curr)
            grad_b.insert(0, db_curr)

        return grad_W, grad_b

    def flushnn(self, filename="ann.json"):
        # uses json format for ease of reading and flushing
        attrs  = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__") and attr != 'layers']
        model = dict()
        for key in attrs:
            val = getattr(self, key)
            if type(val) == np.ndarray:
                val = val.tolist()
            elif type(val) == list:
                val = [v.tolist() if type(v) == np.ndarray else v for v in val]
            model[key] = val

        data = {
            "model": model,
            "layers": []
        }

        for layer in self.layers:
            layer_data = layer.to_json()
            data["layers"].append(layer_data)


        with open(filename, 'w') as fp:
            json.dump(data, fp)

        print(f"neural network flushed to {filename}")

    def loadnn(self, filename):
        f = open(filename, 'r')
        data = json.load(f)

        print(f"loaded neural network from {filename}")
        print(data['model'].keys())
        print(data['layers'][0].keys())
