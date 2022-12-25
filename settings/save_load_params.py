from datetime import datetime
import pickle as pkl
import os
import sys

sys.path.append(os.getcwd() + '/')

from settings import config

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
            os.mkdir(path)
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

    brain_list = []
    with open( inp_path, 'rb' ) as f:
        brain_list=pkl.load(f)
    return brain_list
