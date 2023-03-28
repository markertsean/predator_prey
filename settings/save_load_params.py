from datetime import datetime
import pickle as pkl
import random
import os
import sys

sys.path.append(os.getcwd() + '/')

from settings import config
from perceptron import complex_brain
from perceptron import neural_net

current_date = datetime.today().strftime('%Y.%m.%d.%H.%M.%S')

def get_brain_path(inp_name,use_date):
    global current_date
    path = os.getcwd()+'/data/brains/'+inp_name
    if (path[-1] == '/'):
        path = path[:-1]
    if (use_date):
        path += "_"+current_date+ '/'
    if ( path[-1]!= '/' ):
        path += '/'
    return path

def get_simulation_params():
    return config.simulation_params

def get_character_params():
    for key in config.character_parameters:
        params = config.character_parameters[key]
        if ( ('save_brains' in params) and params['save_brains'] ):
            path = get_brain_path(
                params['brain_output_version'],
                params['brain_output_version_date']
            )
            assert not os.path.exists(path), "Brain path {} already exists!".format(path)
            os.makedirs(path)
    return config.character_parameters

def save_char_brains(box_cell_dict,char_name,char_dict):
    out_path = get_brain_path(
        char_dict['brain_output_version'],
        char_dict['brain_output_version_date']
    )+char_name+".pkl"

    out_list = []
    for cell_number in box_cell_dict:
        cell = box_cell_dict[cell_number]
        for char in cell:
            if ( char.get_name() == char_name ):
                n  = char.get_param("brain").get_n_inputs()
                l  = char.get_param("brain").get_layer_sizes()
                w  = char.get_param("brain").get_weights()
                b  = char.get_param("brain").get_biases()
                af = char.get_param("brain").get_activation_functions()
                od = list(char.get_param("brain_order"))
                out_list.append( (n,l,w,b,af,od) )
                
    with open( out_path, 'wb' ) as f:
        pkl.dump(out_list,f)

def load_char_brains(char_name,brain_version):
    if (brain_version=='latest'):
        brain_version = sorted(
            os.listdir(
                '/'.join(get_brain_path(char_name,False).split('/')[:-2])
            )
        )[-1]

        inp_path = get_brain_path(
            brain_version,
            False
        )+char_name+".pkl"
    else:
        inp_path = os.getcwd()+'/data/brains/'+brain_version

    brain_list = []
    with open( inp_path, 'rb' ) as f:
        brain_list=pkl.load(f)
    return brain_list

def get_val_from_dict(name,this_dict):
    val_dict = {}
    for key in this_dict:
        if name in key:
            val_dict[key] = this_dict[key]
    assert val_dict is not {}, name + " not present in param dict"
    return val_dict

loaded_brain_list = None
def iter_complex_brain_struct( new_item, param_dict ):
    brain_order = []
    brain_objs = []
    brain_dict = {}

    if isinstance( new_item, str ):
        return new_item, new_item, get_val_from_dict( new_item, param_dict )
    
    elif isinstance( new_item, tuple ):
        name, inp_path = new_item

        global loaded_brain_list
        if (loaded_brain_list is None):
            loaded_brain_list = []
            with open( inp_path, 'rb' ) as f:
                brain_list = pkl.load(f)
            for brain in brain_list:
                n,l,w,b,af,od = brain
                loaded_brain_list.append(
                    {
                        "NN":neural_net.NeuralNetwork(
                            n,l,w,b,af
                        ),
                        "order":od,
                    }
                )
        select_dict = random.choice(loaded_brain_list)
        return name, select_dict['NN'], { name: select_dict['order'] }

    elif isinstance( new_item, list ):
        for item in new_item:
            name, objects, val_dict = iter_complex_brain_struct( item, param_dict )
            brain_dict.update(val_dict)
            brain_order.append(name)
            brain_objs.append(objects)

    return brain_order, brain_objs, brain_dict

def load_complex_brains( brain_structure, brain_param_dict ):
    return iter_complex_brain_struct(brain_structure,brain_param_dict)
