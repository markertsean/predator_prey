import random
import numpy as np
import sys
import os

from perceptron import activation

sys.path.append('/'.join( __file__.split('/')[:-2] )+'/')

af_deriv_dict = {
    activation.identity: activation.deriv_identity,
    activation.logistic: activation.deriv_logistic,
    activation.tanh: activation.deriv_tanh,
    activation.relu: activation.deriv_relu,
}

class Neuron:
    def __init__(
        self,
        n_inputs,
        weights=None,
        activation_function=activation.relu,
    ):
        assert isinstance(n_inputs,int)
        self.n_inputs = n_inputs

        if (weights is None):
            self.weights = Neuron.random_weights( self.n_inputs )
        else:
            self.set_weights( weights )

        self.af = activation_function

    def __str__(self,n_indent=1):
        _tabs = max(0,n_indent-1)*"\t"
        tabs  = max(0,n_indent  )*"\t"
        return "{}Neuron:\n{}N inputs: {}\n{}Activation function: {}\n{}Weights: {}\n".format(
            _tabs,
            tabs,self.n_inputs,
            tabs,self.af,
            tabs,self.weights,
        )

    def __eq__(self,other):
        if (
            (self.n_inputs!=other.n_inputs) or
            (self.af      !=other.af      )
        ):
            return False
        for s, o in zip(self.weights,other.weights):
            if (s != o):
                return False
        return True

    def check_numeric(x):
        assert isinstance(x,(float,int))

    def check_weights(x):
        assert isinstance(x,(list,np.ndarray,float,int))

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

    def get_activation(self):
        return self.af

    def get_activation_deriv(self):
        global af_deriv_dict
        return af_deriv_dict[self.af]

    def calc(self,inp_array,bias):
        inp_x = Neuron.set_array(inp_array,self.n_inputs)
        return self.af( float(np.dot(inp_x,self.weights)) + bias )

class Layer:
    def __init__(
        self,
        n_inputs,
        layer_size,
        weights=None,
        bias=None,
        activation_functions=activation.relu,
    ):
        self.af = activation_functions

        assert isinstance(layer_size,int)
        self.layer_size = layer_size

        assert isinstance(n_inputs,int)
        self.n_inputs = n_inputs

        if (bias is None):
            self.bias = self.random_bias()
        else:
            assert isinstance(bias,(int,float))
            self.bias = bias

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

    def __eq__(self,other):
        if (
            (self.n_inputs   != other.n_inputs  ) or
            (self.layer_size != other.layer_size)
        ):
            return False
        for s_node, o_node in zip(self.neuron_list,other.neuron_list):
            if (s_node != o_node):
                return False
        return True

    def get_layer_size(self):
        return self.layer_size

    def get_n_inputs(self):
        return self.n_inputs

    def get_neurons(self):
        return self.neuron_list

    def random_bias(self):
        return random.random()

    def set_bias(self,x):
        Neuron.check_numeric(x)
        self.bias = x

    def get_bias(self):
        return self.bias

    def get_weights(self):
        weights_list = []
        for neuron in self.neuron_list:
            weights_list.append(neuron.get_weights())
        return weights_list

    # Activation functions are the same across a layer
    def get_activation_function(self):
        return self.neuron_list[0].get_activation()

    def get_activation_deriv(self,x):
        global af_deriv_dict
        return af_deriv_dict[self.neuron_list[0].get_activation()](x)

    def calc(self,x):
        y_out = np.zeros(self.layer_size)
        for i in range(0,self.layer_size):
            y_out[i] = self.neuron_list[i].calc(x,self.bias)
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
                    bias                 = layer_biases[i],
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

    def __eq__(self,other):
        if ( self.n_inputs   != other.n_inputs ):
            return False
        if ( len(self.layer_sizes) != len(other.layer_sizes) ):
            return False
        for s_layer, o_layer in zip(self.layer_list,other.layer_list):
            if (s_layer != o_layer):
                return False
        return True

    def get_n_inputs(self):
        return self.n_inputs

    def get_layer_sizes(self):
        return self.layer_sizes

    def get_biases(self):
        bias_list = []
        for layer in self.layer_list:
            bias_list.append(layer.get_bias())
        return bias_list

    def get_weights(self):
        weights_list = []
        for layer in self.layer_list:
            weights_list.append(layer.get_weights())
        return weights_list

    def get_activation_functions(self):
        af_list = []
        for layer in self.layer_list:
            af_list.append(layer.get_activation_function())
        return af_list

    def get_layer(self,n):
        return self.layer_list[n]

    def calc_each_layer_value(self,x):
        x_iter = x
        out_list = []
        for layer in self.layer_list:
            out_list.append(layer.calc(x_iter))
            x_iter = out_list[-1]
        return out_list

    def calc_partial(self,x,n):
        assert n < self.n_layers
        return calc_each_layer_value(x)[n]

    def calc(self,x):
        return self.calc_each_layer_value(x)[-1]

    # Error values: default pred_vals - true_vals
    def backprop(self,input_values,error_values,learning_rate):
        if (isinstance(input_values,list)):
            input_values = np.array(input_values)
        assert isinstance(input_values,np.ndarray)
        if (isinstance(error_values,list)):
            error_values = np.array(error_values)
        assert isinstance(error_values,np.ndarray)
        assert input_values.shape[0]==self.get_n_inputs()

        inp = [ np.atleast_2d(input_values) ]
        for layer in self.calc_each_layer_value(input_values):
            inp.append( np.atleast_2d(layer) )

        W   = []
        B   = []
        for layer in range(len(self.get_weights())):
            layer_weights = self.get_weights()[layer]
            bias = self.get_biases()[layer]
            weights = np.atleast_2d(layer_weights)
            W.append( weights )
            B.append( bias )

        #error = inp[-1] - real_values
        deltas = [ error_values * self.layer_list[-1].get_activation_deriv( inp[-1] ) ]

        for layer in reversed(range(self.n_layers)):
            dw = deltas[-1].dot(W[layer])
            dw = dw * self.layer_list[layer].get_activation_deriv( inp[layer] )
            deltas.append(dw)
        deltas = deltas[::-1]

        for layer in range(self.n_layers):
            W[layer] -= learning_rate * ( (deltas[layer+1].T).dot(inp[layer]) )
            B[layer] -= learning_rate * ( np.average(deltas[layer+1]) )

        for layer in range(self.n_layers):
            new_b = B[layer]
            new_w = W[layer]

            self.layer_list[layer] = Layer(
                n_inputs             = self.layer_list[layer].get_n_inputs(),
                layer_size           = self.layer_list[layer].get_layer_size(),
                weights              = new_w,
                bias                 = new_b,
                activation_functions = self.layer_list[layer].af,
            )
