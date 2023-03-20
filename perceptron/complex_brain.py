#import random
import numpy as np
import sys
#import os

sys.path.append('/'.join( __file__.split('/')[:-2] )+'/')

from perceptron import activation
from perceptron import neural_net

'''
ComplexBrain class allows more complicated analysis of input.
Standard brain is a simple multi layered perceptron. Complex brain takes input
from multiple sources, passes the input through a series of operations
including perceptron or CB operations, then combines these into a 
new multi-layered perceptron. This can allow multiple "brains" to be trained based
on vision alone, and then their inputs combined into one analizing mind. It also
allows factoring of other variables in to the situation, such as the age or remaining
energy of the character, whether it can reproduce, etc.
'''
class ComplexBrain():
    def __init__(
        self
        ,input_names
        ,input_layers
        ,input_operations=None
        ,end_net_layers=None
        ,mid_layer_activations=activation.identity
        ,final_layer_activations=activation.tanh
        ,input_net_cutoffs=-1
        ,nn_inputs=None
        ,mid_operations=None
        ,input_end_network=None
    ):

        self.name_list = input_names
        self.input_operations = input_operations

        assert isinstance( input_layers, list )

        for col in input_layers:
            assert (
                isinstance( col, (str,list,tuple,neural_net.NeuralNetwork,ComplexBrain) ) or
                (col is None)
            ), col+" is of type "+str(type(col))

        '''
        Want to create a chain of functions to implement input from. Operations
        should also be inheriteted from parents,as should ending NN
        '''
        self.input_layers = input_layers
        self.operations = {}
        self.nn_inputs = 0

        if ( (mid_operations is not None) and (nn_inputs is not None) ):
            self.nn_inputs = nn_inputs
            self.operations = mid_operations
        else:
            current_nn_inputs = 0
            for op,name in zip(input_layers,self.name_list):

                if isinstance( op, neural_net.NeuralNetwork ):
                    # Cutoff at whatever layer inserted
                    self.operations[name] = (op.calc_partial,input_net_cutoffs)
                    current_nn_inputs += op.get_layer_sizes()[input_net_cutoffs]
                else:
                    self.operations[name] = activation.identity
                    current_nn_inputs += 1

            self.nn_inputs = current_nn_inputs

        assert (end_net_layers is not None) or (
            (input_end_network is not None) and
            isinstance(input_end_network,neural_net.NeuralNetwork)
        )

        if ( input_end_network is not None ):
            assert self.nn_inputs == input_end_network.get_n_inputs()
            self.comb_NN = input_end_network
        else:
            end_network_afs = []
            for i in range(0,len(end_net_layers)):
                end_network_afs.append(mid_layer_activations)
            end_network_afs[-1]=final_layer_activations

            self.comb_NN = neural_net.NeuralNetwork(
                self.nn_inputs,
                end_net_layers,
                activation_functions=end_network_afs
            )

    def __str__(self,n_indent=1):
        _tabs = max(0,n_indent-1)*"\t"
        tabs  = max(0,n_indent  )*"\t"
        out_str = "{}Input Variable Names:\n".format(
            _tabs
        )
        for name in self.name_list:
            out_str += "{}{}\n".format(
                tabs,
                name
            )
        out_str += "{}Input Operation Structure:\n".format(
            _tabs
        )
        for op in self.input_operations:
            out_str += "{}{}:{}\n".format(
                tabs,
                op,
                self.input_operations[op]
            )
        out_str += "{}Operation Structure:\n".format(
            _tabs
        )
        for op in self.operations:
            out_str += "{}{}:{}\n".format(
                tabs,
                op,
                self.operations[op]
            )
        out_str += "{}Ending Neural Network:\n".format(
            _tabs
        )
        out_str += "{}{}\n".format(
            tabs,
            self.comb_NN
        )
        return out_str

    def set_operations(self,inp_operations):
        self.operations = inp_operations

    def set_NN(self,inp_NN):
        assert isinstance(inp_NN,neural_net.NeuralNetwork)
        assert self.nn_inputs == inp_NN.get_n_inputs()
        self.comb_NN = inp_NN

    def get_input_names(self):
        return self.name_list

    def get_input_layers(self):
        return self.input_layers

    def get_nn_inputs(self):
        return self.nn_inputs

    def get_operations(self):
        return self.operations

    def get_operation_value(self,key,operation=None):
        if operation is None:
            operation = self.input_operations[key]
        if not isinstance(operation,(list,tuple)):
            return operation()
        elif isinstance(operation,tuple):
            list_op = list(operation)
            op = list_op.pop(0)
            tup_op = tuple(list_op)
            return op( *tup_op )
        elif isinstance(operation,list):
            ret_list = []
            for op in operation:
                ret_list.append( self.get_operation_value('',op) )
            return ret_list

    def get_input_values(self,key=None):
        out_dict = {}
        for key in self.input_operations:
            out_dict[key] = self.get_operation_value(key)
        return out_dict

    def get_input_to_final_net(self):
        input_dict = self.get_input_values()

        NN_input = []
        for key in self.name_list:
            input_vals = input_dict[key]
            if ( isinstance(self.operations[key],tuple) ):
                list_op = list(self.operations[key])
                op = list_op.pop(0)
                tup_op = tuple(list_op)
                output_op = op( input_vals, *tup_op )
            else:
                output_op = self.operations[key](input_vals)

            if ( isinstance(output_op,(list,np.ndarray)) ):
                for op in output_op:
                    NN_input.append(op)
            else:
                NN_input.append( output_op )

        return NN_input

    def get_n_inputs(self):
        return self.comb_NN.get_n_inputs()

    def get_layer_size(self):
        return self.comb_NN.get_layer_size()

    def get_layer_sizes(self):
        return self.comb_NN.get_layer_sizes()

    def get_layer_sizes(self):
        return self.comb_NN.get_layer_sizes()

    def get_biases(self):
        return self.comb_NN.get_biases()

    def get_weights(self):
        return self.comb_NN.get_weights()

    def get_activation_functions(self):
        return self.comb_NN.get_activation_functions()

    # Dummy input for compatibility
    def calc(self,dummy_inputs=0):
        NN_input = self.get_input_to_final_net()
        return self.comb_NN.calc( NN_input )

    def backprop(
            self,
            error_values,
            learning_rate
    ):
        input_values = self.get_input_to_final_net()
        self.comb_NN.backprop(
            input_values,
            error_values,
            learning_rate
        )
