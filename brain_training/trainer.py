import pickle as pkl
from datetime import datetime
import numpy as np
import sys
import os

HOMEDIR = '/'.join( __file__.split('/')[:-2] )+'/'
sys.path.append(HOMEDIR)

#from settings import save_load_params
#import characters.parameters as parameters
import perceptron.neural_net as NN
import perceptron.complex_brain as CB
from perceptron import activation

AF_DICT = {
    'identity':activation.identity,
    'logistic':activation.logistic,
    'tanh':activation.tanh,
    'relu':activation.relu,
}

def interpret_conf(val):
    if (val == 'None'):
        return None
    elif (val in AF_DICT):
        return AF_DICT[val]
    elif (val.upper()=="TRUE"):
        return True
    elif (val.upper()=="FALSE"):
        return False
    elif (val[0].isalpha()):
        return val
    elif ('[' in val):
        vals = val.strip('[]').split(',')
        out_list = []
        for v in vals:
            out_list.append(interpret_conf(v))
        return out_list
    elif ('.' in val):
        return float(val)
    elif ('e' in val):
        return float(val)
    elif (val[0].isdigit()):
        return int(val)
    else:
        assert False, val + " is of unknown type!"

def read_conf():
    global HOMEDIR
    out_dict = {}
    with open(HOMEDIR+'/brain_training/trainer.conf','r') as f:
        for line in f:
            l = line.split(':')
            key = l[0].strip()
            val = l[1].strip()
            out_dict[key] = interpret_conf(val)

    return out_dict

def save_nn(inp_nn,param_dict):
    global HOMEDIR
    out_path = HOMEDIR+'data/brains/gen_brain_inputs_{}_layers_{}/'.format(
        param_dict['NN_input_shape'],
        '_'.join([ str(x) for x in param_dict['NN_layers'] ])
    )
    out_fn = param_dict['out_file_name']
    if ( param_dict['append_date'] ):
        out_fn += "_"+datetime.today().strftime('%Y.%m.%d.%H.%M.%S')
    out_full = out_path + out_fn + '.pkl'

    brain_var_list = []
    for key in param_dict:
        if ("brain_var_" in key):
            brain_var_list.append(key)

    var_order_list = []
    for i in range( param_dict['NN_input_shape'] ):
        var_list = []
        for key in sorted( brain_var_list ):
            if ( param_dict[key] == 'i' ):
                var_list.append(i)
            else:
                var_list.append( param_dict[key] )
        var_order_list.append( tuple(var_list) )

    brain_order_list = []
    for tup in var_order_list:
        brain_order_list.append(
            param_dict['brain_order'].format( *tup )
        )

    n  = inp_nn.get_n_inputs()
    l  = inp_nn.get_layer_sizes()
    w  = inp_nn.get_weights()
    b  = inp_nn.get_biases()
    af = inp_nn.get_activation_functions()
    od = brain_order_list
    # Pass list for backwards compatibility
    brain_output = [ (n,l,w,b,af,od) ]

    os.makedirs(out_path,exist_ok=True)
    with open(out_full,'wb') as f:
        pkl.dump(brain_output,f)
    print("Wrote ",out_full)

def train( inp_nn, param_dict ):
    iter_max = param_dict['epochs']
    for i in range( iter_max ):
        inputs = param_dict['NN_min_val'] + (
                param_dict['NN_max_val'] - param_dict['NN_min_val']
            ) * np.random.rand( param_dict['NN_input_shape'] )
        o_err, s_error = (1.,1.)

        inp_nn.backprop(
            inputs,
            [o_err,s_error],
            param_dict['NN_learning_rate']
        )
        if ( ( i % (param_dict['epochs']/100) == 0 ) and (i>0) ):
            print("Trained epoch: {:06}".format(i))

def main():
    in_dict = read_conf()

    print("Running trainer with params:")
    for key in in_dict:
        print("{:40s}\t{}".format(key,in_dict[key]))
    print()

    base_NN = NN.NeuralNetwork(
        in_dict['NN_input_shape'],
        in_dict['NN_layers'],
        in_dict['NN_weights'],
        in_dict['NN_biases'],
        in_dict['NN_layer_afs'],
    )
    train(base_NN,in_dict)

    save_nn( base_NN, in_dict )

if __name__ == '__main__':
    main()
