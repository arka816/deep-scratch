import numpy as np

class Layer:
    def __init__(self, m, n, activation="relu", initializationScheme='randn'):
        self.activation = activation
        self.initializationScheme = initializationScheme
        self.initialize(m, n)
    
    def initialize(self, m, n):
        # VARIOUS INITIALIZATION SCHEMES

        if self.initializationScheme == 'randn':
            # random normal with mean 0
            self.W = np.random.randn(m, n)
            self.b = np.zeros((n, 1))

        elif self.initializationScheme == 'he':
            # He initialization
            self.W = np.random.randn(m, n) * np.sqrt(2 / m)
            self.b = np.zeros((n, 1))

        elif self.initializationScheme == 'glorot-normal':
            # Glorot Normal initialization
            self.W = np.random.randn(m, n) * np.sqrt(2 / (m + n))
            self.b = np.zeros((n, 1))

        elif self.initializationScheme == 'glorot-uniform':
            # Glorot Uniform initialization
            limit = np.sqrt(6/(m + n))
            self.W = np.random.uniform(low = -limit, high = limit, size = (m, n))
            self.b = np.zeros((n, 1))
            
        else:
            raise Exception('Initialization scheme not recognized')

    def to_json(self):
        return {
            "activation": self.activation,
            "initializationScheme": self.initializationScheme,
            "W": self.W.tolist(),
            "b": self.b.tolist()
        }