import perceptron.activation as AF
import math

class char_config:
    def __init__(self):
        self.my_dict = {
            'n': 0
            ,'size':  1e-4

            ,'age': False
            ,'age_max': 10.0

            ,'energy': False
            ,'energy_max':1.0
            ,'energy_speed_delta':0.15 # Per second at max speed
            ,'energy_time_delta':0.10 # Per second

            ,'needs_food': False
            ,'food_source': None
            ,'food_of': None

            ,'vision': False
            ,'eye_offset': 60 * math.pi / 180
            ,'eye_fov': 30 * math.pi / 180
            ,'eye_dist': 0.25
            ,'eye_rays': 5
            ,'eye_objs': []

            ,'brain': False
            ,'brain_layers':[2]
            ,'brain_weights':None
            ,'brain_biases':None
            ,'brain_AF':[AF.tanh]

            ,'complex_brain': False
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

            ,'spawns_fixed': False
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
            ,'self_orientation_reward':1.0
            ,'self_move_reward':0.3
            ,'predator_prey_orientation_reward':1.0
            ,'predator_prey_move_reward':0.3
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

    def __setitem__( self, key, val ):
        assert isinstance( key, str )
        assert ( key in self.my_dict ), key + " not in config class!"
        self.my_dict[key] = val

    def get_dict( self ):
        return self.my_dict
