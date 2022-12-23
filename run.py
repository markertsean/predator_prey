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
import settings.config as cfg

def initialize_characters_homogenous_isotropic(inp_box,inp_char_dict):
    character_class_dict = {
        'food_source': characters.FoodSource,
        'prey': characters.Prey,
    }
    for name in inp_char_dict:
        char_dict = inp_char_dict[name]
        for i in range(char_dict['n']):
            if (name == 'food_source'):
                inp_box.embed(
                    character_class_dict[name](
                        inp_box.get_param('length')*random.random(),
                        inp_box.get_param('length')*random.random(),
                        char_dict['size'],
                    )
                )
            else:
                inp_box.embed(
                    character_class_dict[name](
                        inp_box.get_param('length')*random.random(),
                        inp_box.get_param('length')*random.random(),
                        char_dict['size'],
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
        for char in char_parameters:
            for key in char_parameters[char]:
                f.write("{}_{}:{}\n".format(char,key,char_parameters[char][key]))
    print("Wrote "+input_box.get_param('output_path')+fn)

def main():
    simulation_parameters = cfg.simulation_params
    character_parameters  = cfg.character_parameters

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

if __name__ == "__main__":
    main()
