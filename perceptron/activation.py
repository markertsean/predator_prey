import numpy as np
import math
import sys

sys.path.append('/'.join( __file__.split('/')[:-2] )+'/')

def identity(x):
    return x

def logistic(x):
    return 1. / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return max(0,x)

def deriv_identity(x):
    y=np.copy(x)
    y[True] = 1.
    return y

def deriv_logistic(x):
    s = logistic(x)
    return s * ( 1. - s )

def deriv_tanh(x):
    return (1. - tanh(x)**2)

def deriv_relu(x):
    if hasattr(x, '__iter__'):
        Y = []
        for y in x:
            Y.append(deriv_relu(y))
        if (isinstance(x,np.ndarray)):
            return np.array(Y)
        else:
            return Y
    else:
        if (x>0):
            return 1
        return 0
