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
