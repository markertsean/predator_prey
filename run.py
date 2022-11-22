import random
import sys
import os

sys.path.append(os.getcwd() + '/')

import simulator.simulator as simulator
import characters.characters as characters
import characters.parameters as parameters

def initialize_characters_homogenous_isotropic(initialize_dict,inp_box):
    for key in initialize_dict:
        name, count, size, obj = initialize_dict[key]
        for i in range(0,int(count)):
            inp_box.embed(
                obj(
                    inp_box.get_param('length')*random.random(),
                    inp_box.get_param('length')*random.random(),
                    size,
                    parameters.Speed(
                        inp_box.get_param('max_speed') * 0.1,
                        inp_box.get_param('max_speed')
                    )
                )
            )


def main():
    simulation_parameters = {
        'max_steps':  1e1 ,
        'time_step':  1e-1,
        'box_size':   1e0,
        'cell_size':  1e-1,
        'char_size':  1e-4,
        'n_prey':     5e0 ,
        'seed': 42,
        'abs_max_speed': 1e-2,
        'snapshot_step': 1,
    }

    box = simulator.SimulationBox(
        simulation_parameters['box_size'],
        simulation_parameters['cell_size'],
        simulation_parameters['time_step'],
        simulation_parameters['abs_max_speed'],
        simulation_parameters['snapshot_step'],
        simulation_parameters['seed'],
    )

    initialize_dict = {}
    initialize_dict['prey'] = (
        'prey',
        simulation_parameters['n_prey'],
        simulation_parameters['char_size'],
        characters.Prey
    )
    initialize_characters_homogenous_isotropic(initialize_dict,box)

if __name__ == "__main__":
    #inp_args = __read_args__()
    #run_model_gen_prediction(inp_args)
    main()
