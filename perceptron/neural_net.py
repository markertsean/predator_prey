import random
import numpy as np
import sys
import os

from perceptron import activation

sys.path.append('/'.join( __file__.split('/')[:-2] )+'/')

class Neuron:
    def __init__(
        self,
        n_inputs,
        weights=None,
        bias=None,
        activation_function=activation.relu,
    ):
        assert isinstance(n_inputs,int)
        self.n_inputs = n_inputs

        if (weights is None):
            self.weights = Neuron.random_weights( self.n_inputs )
        else:
            self.set_weights( weights )

        if (bias is None):
            self.bias = Neuron.random_bias()
        else:
            self.set_bias(bias)

        self.af = activation_function

    def __str__(self,n_indent=1):
        _tabs = max(0,n_indent-1)*"\t"
        tabs  = max(0,n_indent  )*"\t"
        return "{}Neuron:\n{}N inputs: {}\n{}Bias: {}\n{}Activation function: {}\n{}Weights: {}\n".format(
            _tabs,
            tabs,self.n_inputs,
            tabs,self.bias,
            tabs,self.af,
            tabs,self.weights,
        )

    def check_numeric(x):
        assert isinstance(x,(float,int))

    def check_weights(x):
        assert isinstance(x,(list,np.ndarray,float,int))

    def random_bias():
        return random.random()

    def set_bias(self,x):
        Neuron.check_numeric(x)
        self.bias = x

    def get_bias(self):
        return self.bias

    def set_array(x,n_inputs):
        Neuron.check_weights(x)
        x_array = np.zeros(n_inputs)
        if ( isinstance(x,float) or isinstance(x,int) ):
            for i in range(0,x_array.shape[0]):
                x_array[i] = x
        else:
            if (isinstance(x,list)):
                n = len(x)
            else:
                n = x.shape[0]
            assert n == n_inputs
            for i in range(0,n):
                x_array[i] = x[i]
        return x_array

    def random_weights(n):
        weights = np.zeros(n)
        for i in range(0,n):
            weights[i] = 2 * ( random.random() - 0.5 )
        return weights

    def set_weights(self,x):
        self.weights = Neuron.set_array(x,self.n_inputs)

    def get_weights(self):
        return self.weights

    def calc(self,inp_array):
        inp_x = Neuron.set_array(inp_array,self.n_inputs)
        return self.af( float(np.dot(inp_x,self.weights)) + self.bias )

class Layer:
    def __init__(
        self,
        n_inputs,
        layer_size,
        weights=None,
        biases=None,
        activation_functions=activation.relu,
    ):
        self.af = activation_functions

        assert isinstance(layer_size,int)
        self.layer_size = layer_size

        assert isinstance(n_inputs,int)
        self.n_inputs = n_inputs

        bias_list = []
        if (biases is None):
            for i in range(0,self.layer_size):
                bias_list.append( Neuron.random_bias() )
        else:
            Neuron.check_weights( biases )
            if isinstance(biases,(int,float)):
                for i in range(0,self.layer_size):
                    bias_list.append( biases )
            else:
                bias_list = Neuron.set_array( biases, self.n_inputs )

        weight_list = []
        if (weights is None):
            for i in range(0,self.layer_size):
                weight_list.append( Neuron.random_weights(self.n_inputs) )
        else:
            Neuron.check_weights(weights)
            if ( isinstance(weights,(list,np.ndarray)) ):
                for row in weights:
                    weight_row = Neuron.set_array( row, self.n_inputs )
                    weight_list.append( weight_row )
            else:
                row = []
                for i in range( 0, self.n_inputs ):
                    row.append( weights )
                weight_row = Neuron.set_array( row, self.n_inputs )
                for i in range( 0, self.layer_size ):
                    weight_list.append( weight_row )


        self.neuron_list = []
        for i in range(0,self.layer_size):
            self.neuron_list.append(
                Neuron(
                    self.n_inputs,
                    weight_list[i],
                    bias_list[i],
                    self.af
                )
            )

    def __str__(self,n_indent=1):
        _tabs = max(0,n_indent-1)*"\t"
        tabs  = max(0,n_indent  )*"\t"
        out_str = "{}Layer:\n{}Layer Size: {}\n".format(
            _tabs,
            tabs,self.layer_size,
        )
        for neuron in self.neuron_list:
            out_str = out_str + neuron.__str__(n_indent+1)
        return out_str

    def get_layer_size(self):
        return self.layer_size

    def get_n_inputs(self):
        return self.n_inputs

    def get_neurons(self):
        return self.neuron_list

    def calc(self,x):
        y_out = np.zeros(self.layer_size)
        for i in range(0,self.layer_size):
            y_out[i] = self.neuron_list[i].calc(x)
        return y_out

class NeuralNetwork:
    def __init__(
        self,
        n_inputs_init,
        layer_sizes,
        weights=None,
        biases=None,
        activation_functions=activation.relu,
    ):
        assert isinstance(n_inputs_init,int)
        self.n_inputs = n_inputs_init

        assert isinstance(layer_sizes,list)
        assert len(layer_sizes)>0
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)

        layer_weights = []
        if (weights is not None):
            assert isinstance(weights,(int,float,list))
        if (isinstance(weights,list)):
            assert len(weights)==self.n_layers
            layer_weights = weights
        else:
            for i in range(0,self.n_layers):
                layer_weights.append(weights)

        layer_biases = []
        if (biases is not None):
            assert isinstance(biases,(int,float,list))
        if (isinstance(biases,list)):
            assert len(biases)==self.n_layers
            layer_biases = biases
        else:
            for i in range(0,self.n_layers):
                layer_biases.append(biases)

        layer_af = []
        if (isinstance(activation_functions,list)):
            assert len(activation_functions)==self.n_layers
            layer_af = activation_functions
        else:
            for i in range(0,self.n_layers):
                layer_af.append(activation_functions)

        self.layer_list = []
        n_inputs = self.n_inputs
        for i in range(0,self.n_layers):
            self.layer_list.append(
                Layer(
                    n_inputs,
                    layer_size           = layer_sizes[i],
                    weights              = layer_weights[i],
                    biases               = layer_biases[i],
                    activation_functions = layer_af[i],
                )
            )
            n_inputs = layer_sizes[i]

    def __str__(self,n_indent=1):
        _tabs = max(0,n_indent-1)*"\t"
        tabs  = max(0,n_indent  )*"\t"
        out_str = "{}Neural Network:\n".format(
            _tabs,
            tabs,
        )
        for layer in self.layer_list:
            out_str = out_str + layer.__str__(n_indent+1)
        return out_str

    def calc(self,x):
        x_iter = x
        i = 0
        for layer in self.layer_list:
            i = i+1
            x_iter = layer.calc(x_iter)
        return x_iter
