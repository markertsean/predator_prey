#import random
#import numpy as np
import sys
#import os

sys.path.append('/'.join( __file__.split('/')[:-2] )+'/')

from perceptron import activation
from perceptron import neural_net

class ComplexBrain:
    def __init__(
        self
        #,input_shape
        ,input_layers
        ,mid_layer_activations=activation.identity
        ,final_layer_activations=activation.tanh
    ):

        #assert isinstance( input_shape, (list,tuple) )
        assert isinstance( input_layers, list )
        
        #for col in input_shape:
        #    assert isinstance( col, int )

        for col in input_layers:
            assert (
                isinstance( col, (int,list,tuple,neural_net.NeuralNetwork,ComplexBrain) ) or
                (col is None)
            )

        '''
        At the end we want a neural network to combine all the inputs, and for training
        Pop this off the end of the list for separate handling
        '''
        end_network_layers = []
        end_network_afs    = []
        for i in range(len(input_layers)-1,-1,-1):
            col = input_layers[i]
            if (isinstance(col,int)):
                end_network_layers.append(col)
                end_network_afs.append(mid_layer_activations)
                input_layers.pop(i)
            else:
                break
        end_network_layers = end_network_layers[::-1]
        end_network_afs[-1] = final_layer_activations

        assert len(end_network_layers)>0, "Complex brain requires end network"

        '''
        Want to create a chain of functions to implement input from
        '''
        self.name_list = []
        self.operations = []
        self.layer_outputs = []
        self.nn_inputs = 0

        for i in range(0,len(input_layers)):
            column = input_layers[i]

            new_operations = []
            new_layer_out  = []
            
            current_nn_inputs = 0

            if not isinstance( column, list ):
                column = [column]

            for col in column:

                if ( i==0 ):
                    assert isinstance(col,tuple)
                    name, c = col
                    self.name_list.append(name)
                else:
                    c = col
                
                if isinstance( c, neural_net.NeuralNetwork ):
                    new_operations.append( (c.calc_partial,-2,) )
                    # Not taking the final layer, but one before
                    new_layer_out.append( c.get_layer_sizes()[-2] )
                if isinstance( c, int ) or (c is None):
                    new_operations.append( activation.identity )
                    new_layer_out.append( 1 )

                current_nn_inputs += new_layer_out[-1]

            self.operations.append( new_operations )
            self.layer_outputs.append( new_layer_out )
            self.nn_inputs = current_nn_inputs

        self.comb_NN = neural_net.NeuralNetwork(
            self.nn_inputs,
            end_network_layers,
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
        out_str += "{}Layer Structure:\n".format(
            _tabs
        )
        for l in self.layer_outputs:
            out_str += "{}{}\n".format(
                tabs,
                l
            )
        out_str += "{}Operation Structure:\n".format(
            _tabs
        )
        for op in self.operations:
            out_str += "{}{}\n".format(
                tabs,
                op
            )
        out_str += "{}Ending Neural Network:\n".format(
            _tabs
        )
        out_str += "{}{}\n".format(
            tabs,
            self.comb_NN
        )
        return out_str

    def get_input_names(self):
        return self.name_list

    def calc(self,inputs):

        # Ensure input is dict object of name/value matching inputs
        assert isinstance(inputs,(list,dict,))

        input_dict = inputs
        if isinstance( inputs, list ):
            input_dict = {}
            for col in inputs:
                assert isinstance(col,tuple)
                name, val = col
                input_dict[name] = val

        layer_input = []
        for name in self.name_list:
            if name not in input_dict:
                print(name," not in ComplexBrain calc input!")
                assert False

            layer_input.append(input_dict[name])

        # Create list of layer inputs
        all_layer_inputs = [layer_input]
        layer_input = []

        for operations in self.operations:
            if ( not isinstance(operations,list) ):
                operations = [operations]
            
            for i in range(0,len(operations)):
                operation = operations[i]
                x = all_layer_inputs[-1][i]
                if (isinstance(operation,tuple)):
                    op, arg = operation
                    layer_input.append( op( x, arg ) )
                else:
                    layer_input.append( operation( x ) )

            all_layer_inputs.append(layer_input)
            layer_input = []

        final_layer_output = []
        for val in all_layer_inputs[-1]:
            if isinstance(val,(int,float,)):
                final_layer_output.append(val)
            else:
                for v in val:
                    final_layer_output.append(v)

        return self.comb_NN.calc( final_layer_output )

CB = ComplexBrain(
    [
        [
            ("A",neural_net.NeuralNetwork(3,[5,4]),),
            ("B",neural_net.NeuralNetwork(2,[8,7]),),
            ("C",2,),
            ("D",3,),
        ]
        ,2
        ,3
    ]
)

print(CB)

print(
    CB.calc(
        [
            ("D",1,),
            ("C",2,),
            ("A",[1,2,3],),
            ("B",[4,5],),
        ]
    )
)
