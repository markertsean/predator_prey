import perceptron.activation as AF
import math
import sys
import os

sys.path.append(os.getcwd() + '/')

simulation_params = {
    'max_steps':        int(1e2)
    ,'time_step':           1e-1
    ,'snapshot_step':   int(1e0)
    
    ,'box_size':            1e0
    ,'cell_size':           1e-1
    ,'abs_max_speed':     2.5e-2

    ,'max_characters':  int(5e2)
    ,'kill_no_diff':        True

    # Parallelization currently not working
    ,'n_jobs':                 1
    ,'parallel_char_min':   None#int(3e1),

    ,'seed':                None
}

food_params = {
    'n': 40
    ,'size': simulation_params['cell_size']/2.
}

prey_params = {
    'n': 50
    ,'size':  food_params['size'] / 10.

    ,'age': True
    ,'age_max':10000.0

    ,'energy': True
    ,'energy_max':1.0
    ,'energy_speed_delta':0.15 # Per second at max speed
    ,'energy_time_delta':0.10 # Per second

    ,'needs_food': True
    ,'food_source': 'food_source'

    ,'vision': True
    ,'eye_offset': 60 * math.pi / 180
    ,'eye_fov': 30 * math.pi / 180
    ,'eye_dist': 0.25
    ,'eye_rays': 5

    ,'brain': True
    ,'brain_layers':[7,2]
    ,'brain_weights':None
    ,'brain_biases':None
    ,'brain_AF':[AF.relu,AF.tanh]

    ,'complex_brain': True
    ,'complex_brain_input_structure': [
        'age',
        'energy',
        ('eyes_food_source','data/brains/test_2023.03.10.13.11.50/prey.pkl',),
    ]
    ,'complex_brain_nn_structure': [
        5,
        5,
        2
    ]
    ,'complex_brain_variables':[
        'age',
        'energy',
        'eyes_food_source',
    ]

    ,'spawns_fixed': True
    ,'spawn_time_fixed': 5.0
    ,'spawns_proba': False
    ,'spawn_prob_sec': 0.0
    ,'new_spawn_delay': 3.0
    ,'spawn_energy_min': 0.4
    ,'spawn_energy_delta': 0.3
    ,'mutation_max':1e-1
    ,'mutation_floor':1e-2
    ,'mutation_halflife':1e1 # generations

    ,'learns': False
    ,'food_source_orientation_reward':1.0
    ,'food_source_move_reward':0.3
    ,'food_source_enter_reward':1.0
    ,'feeds_reward':0.5
    ,'learning_max':1e-1
    ,'learning_floor':1e-2
    ,'learning_halflife':5.0 # age units

    ,'save_brains': False
    ,'brain_output_version': "test"
    ,'brain_output_version_date': True

    ,'load_brains': False
    ,'brain_input_version': "latest"
}

character_parameters = {
    'food_source':food_params,
    'prey':prey_params,
}
