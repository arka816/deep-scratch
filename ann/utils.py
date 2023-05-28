import numpy as np

def clip(a, v):
    '''
        clip(a, v) = min{max{a, -v}, v}
    '''
    return np.minimum(np.maximum(a, -v), v)
