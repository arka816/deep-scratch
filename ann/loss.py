import numpy as np

def binary_crossentropy(y_pred, y_train):
    # CALCULATES THE BINARY CROSS-ENTROPY LOSS 
    samples = y_pred.shape[1]
    loss = -1 / samples * (np.dot(y_train, np.log(y_pred + 1e-10).T) + np.dot((1-y_train), np.log(1-y_pred + 1e-10).T))
    return np.squeeze(loss)

def mse(y_pred, y_train):
    # CALCULATES THE MEAN SQUARE LOSS
    samples = y_pred.shape[1]
    loss = 1 / samples * np.sum((y_pred - y_train) ** 2)
    return np.squeeze(loss)

def cat_crossentropy(y_pred, y_train):
    # CALCULATES THE CATEGORICAL CROSS ENTROPY IN MULTI CLASS CLASSIFICATION
    samples = y_pred.shape[1]
    return -np.sum(y_train * np.log(y_pred)) / samples


def d_binary_crossentropy(y_pred, y_train):
    # calculates the derivative of binary crossentropy
    return - (np.divide(y_train, y_pred + 1e-10) - np.divide(1-y_train, 1-y_pred + 1e-10))

def d_mse(y_pred, y_train):
    # calculates the derivative of mse
    return 2 * (y_pred - y_train)

def d_cat_crossentropy(y_pred, y_train):
    # calculates the derivative of categorical crossentropy
    return - np.divide(y_train, y_pred + 1e-10)