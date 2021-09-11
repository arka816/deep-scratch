# DIFFERENT ACTIVATION FUNCTIONS FOR NEURAL NETWORKS
import numpy as np

def sigmoid(z):
    z = np.where(z < -700, -700, z)
    return 1 / (1 + np.exp(-z))

def tanh(z):
    return np.tanh(z)

def ReLU(z):
    return np.maximum(z, 0)

def leaky_ReLU(z, neg_slope = 0.01):
    return np.where(z > 0, z, neg_slope * z)

def softmax(z):
    e_z = np.exp(z - np.max(z)).astype(np.float64)
    return e_z/np.sum(e_z, axis = 0)

def d_sigmoid(z):
    sig = sigmoid(z)
    return sig * (1-sig)

def d_tanh(z):
    return 1 - np.square(tanh(z))

def d_relu(z):
    return np.where(z >= 0, 1, 0)

def d_leaky_relu(z, neg_slope = 0.01):
    return np.where(z >= 0, 1, neg_slope)

def d_softmax(z):
    s = softmax(z)
    return s * (1-s)