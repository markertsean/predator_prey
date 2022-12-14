import numpy as np
import math
import sys

sys.path.append('/'.join( __file__.split('/')[:-2] )+'/')

def identity(x):
    return x

def logistic(x):
    return 1. / (1 + np.exp(-x))

def tanh(x):
    return math.tanh(x)

def relu(x):
    return max(0,x)

def deriv_identity(x):
    return 1

def deriv_logistic(x):
    return np.exp(-x) * (logistic(x)**2)

def deriv_tanh(x):
    return (1. - math.tanh(x)**2)

def deriv_relu(x):
    if (x>0):
        return 1
    return 0
