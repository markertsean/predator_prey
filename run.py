import random
import math
import numpy as np
import sys
import os
import copy

sys.path.append(os.getcwd() + '/')

import simulator.simulator as simulator
import characters.characters as characters
import characters.parameters as parameters
import perceptron.neural_net as NN
import settings.save_load_params as slp

def get_random_brain(inp_brain_list):
    n_tot = len(inp_brain_list)
    n_bin = int(random.random()*n_tot)
    n_input, layer_sizes, weights , biases, afs, order = inp_brain_list[n_bin]
    brain = NN.NeuralNetwork(
        n_inputs_init        = n_input,
        layer_sizes          = layer_sizes,
        weights              = weights,
        biases               = biases,
        activation_functions = afs,
    )
    return brain, order

def initialize_characters_homogenous_isotropic(inp_box,inp_char_dict):
    character_class_dict = {
        'food_source': characters.FoodSource,
        'prey': characters.Prey,
        'predator': characters.Predator,
    }
    for name in inp_char_dict:
        char_dict = inp_char_dict[name]
        embed_list = []

        brain_list = None
        if (
            ('load_brains' in char_dict) and
            char_dict['load_brains']
        ):
            brain_list = slp.load_char_brains(name,char_dict['brain_input_version'])

        for i in range(char_dict['n']):
            if (name == 'food_source'):
                embed_list.append(
                    character_class_dict[name](
                        inp_box.get_param('length')*random.random(),
                        inp_box.get_param('length')*random.random(),
                        char_dict['size'],
                    )
                )
            else:
                new_char = character_class_dict[name](
                    inp_box.get_param('length')*random.random(),
                    inp_box.get_param('length')*random.random(),
                    char_dict['size'],
                    parameters.Speed(
                        inp_box.get_param('max_speed') * 0.1,
                        inp_box.get_param('max_speed')
                    ),
                    input_parameters=char_dict,
                )
                if ( brain_list is not None ):
                    new_brain, new_brain_order = get_random_brain(brain_list)
                    new_char.set_param( 'brain'      , new_brain )
                    new_char.set_param( 'brain_order', new_brain_order )
                embed_list.append(new_char)

        for char in embed_list:
            inp_box.embed( char )

def save_setup_logfile( sim_parameters, char_parameters, input_box ):
    out_path = input_box.get_param('output_path')+"logfiles/"+input_box.get_param('output_version')+"/"
    os.makedirs(out_path,exist_ok=True)
    fn = "setup.log"
    with open(out_path+fn,'w') as f:
        for key in sim_parameters:
            f.write("{}:{}\n".format(key,sim_parameters[key]))
        for char in char_parameters:
            for key in char_parameters[char]:
                f.write("{}_{}:{}\n".format(char,key,char_parameters[char][key]))
    print("Wrote "+input_box.get_param('output_path')+fn)

def main():
    simulation_parameters = slp.get_simulation_params()
    character_parameters  = slp.get_character_params()

    box = simulator.SimulationBox(
        simulation_parameters['box_size'],
        simulation_parameters['cell_size'],
        int(simulation_parameters['max_steps']),
        simulation_parameters['time_step'],
        simulation_parameters['abs_max_speed'],
        simulation_parameters['snapshot_step'],
        simulation_parameters['max_characters'],
        simulation_parameters['seed'],
        n_jobs = simulation_parameters['n_jobs'],
        parallel_char_min = simulation_parameters['parallel_char_min'],
        kill_early = simulation_parameters['kill_no_diff'],
    )

    save_setup_logfile( simulation_parameters, character_parameters, box )

    initialize_characters_homogenous_isotropic(box,character_parameters)

    box.run_simulation()

    for char_type in character_parameters:
        char_dict = character_parameters[char_type]
        if (
            ('save_brains' in char_dict) and
            char_dict['save_brains']
        ):
            slp.save_char_brains(box.cell_dict,char_type,char_dict)

if __name__ == "__main__":
    main()
