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
import simulation_cfg
import character_cfg

def initialize_characters_homogenous_isotropic(initialize_dict,inp_box,char_dict):
    for key in initialize_dict:
        name, count, size, obj = initialize_dict[key]
        for i in range(0,int(count)):
            if (name == 'food source'):
                inp_box.embed(
                    obj(
                        inp_box.get_param('length')*random.random(),
                        inp_box.get_param('length')*random.random(),
                        size,
                    )
                )
            else:
                inp_box.embed(
                    obj(
                        inp_box.get_param('length')*random.random(),
                        inp_box.get_param('length')*random.random(),
                        size,
                        parameters.Speed(
                            inp_box.get_param('max_speed') * 0.1,
                            inp_box.get_param('max_speed')
                        ),
                        input_parameters=char_dict,
                    )
                )

def save_setup_logfile( sim_parameters, char_parameters, input_box ):
    os.makedirs(input_box.get_param('output_path'),exist_ok=True)
    fn = "setup.log"
    with open(input_box.get_param('output_path')+fn,'w') as f:
        for key in sim_parameters:
            f.write("{}:{}\n".format(key,sim_parameters[key]))
        for key in char_parameters:
            f.write("{}:{}\n".format(key,char_parameters[key]))
    print("Wrote "+input_box.get_param('output_path')+fn)

def main():
    simulation_parameters = simulation_cfg.parameters
    character_parameters  = character_cfg.parameters

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

    initialize_dict = {}
    initialize_dict['food'] = (
        'food source',
        character_parameters['n_food'],
        character_parameters['food_size'],
        characters.FoodSource
    )
    initialize_dict['prey'] = (
        'prey',
        character_parameters['n_prey'],
        character_parameters['prey_size'],
        characters.Prey
    )
    
    initialize_characters_homogenous_isotropic(initialize_dict,box,character_parameters)

    box.run_simulation()
    #foo = characters.Prey(
    #    box.get_param('length')*random.random(),
    #    box.get_param('length')*random.random(),
    #    character_parameters['prey_size'],
    #    parameters.Speed(
    #        box.get_param('max_speed') * 0.1,
    #        box.get_param('max_speed')
    #    ),
    #    input_parameters=character_parameters,
    #)
    #for w in foo.get_param('brain').get_weights():
    #    print(w)
    #
    #dummy_errors = np.zeros(2)
    #dummy_errors[0] = 1.0
    #dummy_errors[1] = -1.0
    #
    #dummy_inputs = np.zeros(22)+1
    #
    #for i in range(0,10):
    #    foo.get_param('brain').backprop(dummy_inputs,dummy_errors,0.1)
    #
    #print('---------------')
    #for w in foo.get_param('brain').get_weights():
    #    print(w)

if __name__ == "__main__":
    main()
