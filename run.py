import random
import math
import numpy as np
import sys
import os

sys.path.append(os.getcwd() + '/')

import simulator.simulator as simulator
import characters.characters as characters
import characters.parameters as parameters

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
    simulation_parameters = {
        'max_steps':  1e2 ,
        'time_step':  1e-1,
        'box_size':   1e0,
        'cell_size':  1.,#1e-1,
        'seed': 43,
        'abs_max_speed': 1e-1,
        'snapshot_step': 1,
        
    }

    character_parameters = {
        'n_food': 10,
        'food_size': simulation_parameters['box_size']*3e-1,

        'n_prey': 5,
        'prey_size':  1e-2,

        'prey_age': False,
        'prey_age_max':10.0,

        'prey_energy': False,
        'prey_energy_max':1.0,
        'prey_energy_speed_delta':0.25, # Per second at max speed
        'prey_energy_time_delta':0.25, # Per second

        'prey_needs_food': False,
        'prey_food_source': 'food source',


        'prey_vision': True,
        'prey_eye_offset': math.pi/2.,#30 * math.pi / 180,
        'prey_eye_fov': math.pi,#30 * math.pi / 180,
        'prey_eye_dist': 0.5,#1e-1,
        'prey_eye_rays': 5,

        'prey_spawns_fixed': False,
        'prey_spawn_time_fixed': 1.0,
        'prey_spawns_proba': False,
        'prey_spawn_prob_sec': 'food source',

        'prey_new_spawn_delay': 1.0,
        'prey_spawn_energy_min': 0.5,
        'prey_spawn_energy_delta': 0.1,

    }

    box = simulator.SimulationBox(
        simulation_parameters['box_size'],
        simulation_parameters['cell_size'],
        int(simulation_parameters['max_steps']),
        simulation_parameters['time_step'],
        simulation_parameters['abs_max_speed'],
        simulation_parameters['snapshot_step'],
        simulation_parameters['seed'],
    )

    save_setup_logfile( simulation_parameters, character_parameters, box )

    #initialize_dict = {}
    #initialize_dict['food'] = (
    #    'food source',
    #    character_parameters['n_food'],
    #    character_parameters['food_size'],
    #    characters.FoodSource
    #)
    #initialize_dict['prey'] = (
    #    'prey',
    #    character_parameters['n_prey'],
    #    character_parameters['prey_size'],
    #    characters.Prey
    #)
    #initialize_characters_homogenous_isotropic(initialize_dict,box,character_parameters)

    box.embed(
        characters.FoodSource(
            0.25,
            0.50,
            0.25,
        )
    )
    box.embed(
        characters.FoodSource(
            0.75,
            0.50,
            0.25,
        )
    )
    box.embed(
        characters.FoodSource(
            0.6 ,
            0.5 ,
            0.05,
        )
    )
    box.embed(
        characters.Prey(
            0.5,
            0.1,
            1e-1,
            parameters.Speed(
                0.1,
                0.1,
                0.05
            ),
            orientation = math.pi/2,
            input_parameters=character_parameters,
        )
    )
    box.run_simulation()

if __name__ == "__main__":
    #inp_args = __read_args__()
    #run_model_gen_prediction(inp_args)
    main()
