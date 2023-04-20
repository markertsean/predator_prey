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
    y=np.copy(x)
    ind = y>0
    y[True] = 0.
    y[ind] = 1.
    return y
