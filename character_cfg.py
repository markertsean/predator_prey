import perceptron.activation as AF
import math
import sys
import os

sys.path.append(os.getcwd() + '/')


parameters = {
    'n_food': 10,
    'food_size': 2e-1,

    'n_prey': 50,
    'prey_size':  1e-3,

    'prey_age': True,
    'prey_age_max':30.0,

    'prey_energy': True,
    'prey_energy_max':1.0,
    'prey_energy_speed_delta':0.15, # Per second at max speed
    'prey_energy_time_delta':0.05, # Per second

    'prey_needs_food': True,
    'prey_food_source': 'food source',

    'prey_vision': True,
    'prey_eye_offset': 30 * math.pi / 180,
    'prey_eye_fov': 60 * math.pi / 180,
    'prey_eye_dist': 5e-2,
    'prey_eye_rays': 5,

    'prey_brain': True,
    'prey_brain_layers':[2],
    'prey_brain_weights':None,
    'prey_brain_biases':None,
    'prey_brain_AF':[AF.tanh],
    'prey_mutation_max':1e-1,
    'prey_mutation_floor':1e-2,
    'prey_mutation_halflife':1e1, # generations

    'prey_spawns_fixed': True,
    'prey_spawn_time_fixed': 5.0,
    'prey_spawns_proba': False,
    'prey_spawn_prob_sec': 'food source',

    'prey_new_spawn_delay': 0.5,
    'prey_spawn_energy_min': 0.4,
    'prey_spawn_energy_delta': 0.1,
}
