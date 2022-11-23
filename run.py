import random
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

def save_setup_logfile( input_parameters, input_box ):
    os.makedirs(input_box.get_param('output_path'),exist_ok=True)
    fn = "setup.log"
    with open(input_box.get_param('output_path')+fn,'w') as f:
        for key in input_parameters:
            f.write("{}:{}\n".format(key,input_parameters[key]))
    print("Wrote "+input_box.get_param('output_path')+fn)

def main():
    simulation_parameters = {
        'max_steps':  1e2 ,
        'time_step':  1e-1,
        'box_size':   1e0,
        'cell_size':  1e-1,
        'seed': 42,
        'abs_max_speed': 1e-1,
        'snapshot_step': 1,
        
    }

    character_parameters = {
        'n_prey': 5,
        'prey_size':  1e-4,

        'prey_age': False,
        'prey_age_max':5.0,

        'prey_energy': False,
        'prey_energy_max':1.0,
        'prey_energy_speed_delta':0.25, # Per second at max speed
        'prey_energy_time_delta':0.25, # Per second
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

    save_setup_logfile( simulation_parameters, box )

    initialize_dict = {}
    initialize_dict['prey'] = (
        'prey',
        character_parameters['n_prey'],
        character_parameters['prey_size'],
        characters.Prey
    )
    initialize_characters_homogenous_isotropic(initialize_dict,box,character_parameters)

    box.run_simulation()


if __name__ == "__main__":
    #inp_args = __read_args__()
    #run_model_gen_prediction(inp_args)
    main()
