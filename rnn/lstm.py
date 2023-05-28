import numpy as np
import sys
sys.path.insert(1, 'C:/github/deep-scratch/ann/')
import matplotlib.pyplot as plt
from activations import sigmoid, tanh, d_sigmoid, d_tanh, softmax


class LSTM:
    def __init__(self):
        pass

    def one_step_forward_pass(self, c_prev, h_prev, x):
        # c_prev : 
        # h_prev : hidden state vector (n_h x 1) from previous lstm
        # x      : current input vector

        z = np.row_stack(h_prev, x)

        # dot product
        f = sigmoid(np.dot(self.Wf, z) + self.bf)
        i = sigmoid(np.dot(self.Wi, z) + self.bi)
        c_bar = tanh(np.dot(self.Wc, z) + self.bc)
        o = sigmoid(np.dot(self.Wo, z) + self.bo)

        # hadamard product
        c = f * c_prev + i * c_bar
        h = o * tanh(c)

        v = np.dot(self.Wv, h) + self.bv
        y_pred = softmax(v)
        return y_pred, v, h, o, c, c_bar, i, f, z

    def forward_backward(self, x, y, h_prev, c_prev):
        for t in range(self.seq_len):
            pass
