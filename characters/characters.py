import random
import math
import numpy as np
import sys
import os
import copy

sys.path.append('/'.join( __file__.split('/')[:-2] )+'/')

from settings import save_load_params
import characters.parameters as parameters
import perceptron.neural_net as NN
import perceptron.complex_brain as CB

class Character:
    id_num = 0
    name = 'Generic Character'
    def __init__(
        self,
        x_init,
        y_init,
        size,
        speed,
        orientation = None,
        parent = -1,
        input_parameters = {},
    ):
        self.collision = False
        self.id   = Character.id_num
        Character.id_num += 1

        assert isinstance(x_init,float)
        assert isinstance(y_init,float)
        assert isinstance(size  ,float)

        self.x    = x_init
        self.y    = y_init
        self.size = size
        self.radius = self.size/2.
        self.name = Character.name
        self.generation = 0

        assert isinstance(speed,parameters.Speed)
        self.speed = speed

        if (orientation is None):
            self.orientation = parameters.Orientation(value=2*math.pi*random.random())
        elif (isinstance(orientation,float)):
            self.orientation = parameters.Orientation(value=orientation)
        elif (isinstance(orientation,parameters.Orientation)):
            self.orientation = parameters.Orientation
        else:
            raise ValueError("orientation must be None, float, or Orientation object")

        assert isinstance(parent,int)
        self.parent = parent

        self.consumed=False

        self.__setup__(input_parameters)

    def __setup__(self,input_parameters):
        self.input_parameters = input_parameters

    def __str__(self):
        this_dict = self.__dict__
        out_str = str(self.get_name()) + ":\n"
        for key in this_dict:
            if (
                (key!='name') and
                (
                    isinstance(this_dict[key],str) or
                    isinstance(this_dict[key],int) or
                    isinstance(this_dict[key],float)
                )
            ):
                out_str += "\t"+key+":\t"+str(this_dict[key])+"\n"
        for key in this_dict:
            if (
                (key!='name') and
                not (
                    isinstance(this_dict[key],str) or
                    isinstance(this_dict[key],int) or
                    isinstance(this_dict[key],float)
                )
            ):
                out_str += str(this_dict[key])+"\n"
        out_str += 30*'-'+'\n'
        return out_str

    def __additional_equal__(self,other):
        return True

    def __eq__(self,other):
        for param in [
                'name',
                'collision',
                'x',
                'y',
                'speed',
                'size',
                'radius',
                'orientation',
        ]:
            print(param)
            if (
                (param in  self.list_params()) !=
                (param in other.list_params())
            ):
                return False
            print(param)
            if (
                (param in  self.list_params()) and
                (param in other.list_params())
            ):
                if ( self.get_param(param)!= other.get_param(param) ):
                    return False
        if (not self.__additional_equal__(other)):
            return False
        return True


    def update_pos(self,x,y):
        self.x = x
        self.y = y

    def __update_speed__(self):
        pass

    def __update_orientation__(self):
        pass

    def update(self):
        pass

    def list_params(self):
        return self.__dict__.keys()

    def get_param(self,name):
        assert name in self.__dict__
        return self.__dict__[name]

    def set_param(self,name,value):
        assert name in self.__dict__
        self.__dict__[name] = value

    def get_name(self):
        return None

    def get_pos(self):
        return self.x, self.y

    def get_speed(self):
        if ('speed' not in self.list_params()):
            return 0.0
        return self.speed.value

    def get_orientation(self):
        if ('orientation' not in self.list_params()):
            return -1.0
        return self.orientation.value

    def get_age(self):
        if ('age' not in self.list_params()):
            return -1.0
        if (self.age is None):
            return -1.0
        return self.age.value

    def get_energy(self):
        if ('energy' not in self.list_params()):
            return -1.0
        if (self.energy is None):
            return -1.0
        return self.energy.value

    def age_character(self,time_step):
        if (self.age is None):
            return True
        if (self.age.value >= self.age.get_param('max')):
            return False
        self.age.update(time_step)
        return True

    def use_energy(self,time_step):
        if (self.energy is None):
            return True
        if (self.energy.value <= self.energy.get_param('min')):
            return False
        self.energy.update(time_step,self.speed.value)
        return True

    def eat(self):
        pass

    def percieve(self):
        pass

    def spawn(self):
        return False

    def get_pickle_obj(self):
        return []

class FoodSource(Character):
    def __init__(
        self,
        x_init,
        y_init,
        size,
    ):
        self.collision = False
        self.id   = Character.id_num
        Character.id_num += 1

        assert isinstance(x_init,float)
        assert isinstance(y_init,float)
        assert isinstance(size  ,float)

        self.x    = x_init
        self.y    = y_init
        self.size = size
        self.radius = self.size/2.
        self.name = 'food_source'
        self.consumed=False

    def get_name(self):
        return self.name

    def get_pickle_obj(self):
        return [
            ('id'       ,self.id       ),
            ('x'        ,self.x        ),
            ('y'        ,self.y        ),
            ('size'     ,self.size     ),
            ('radius'   ,self.radius   ),
            ('name'     ,self.name     ),
            ('consumed' ,self.consumed ),
            ('collision',self.collision),
        ]

class Prey(Character):
    def get_name(self):
        self.name = 'prey'
        return 'prey'

    def check_param(self,key,param):
        return (key in param) and param[key]

    def __setup__(self,params):
        self.input_parameters = params

        self.collision=True
        self.consumed=True

        brain_param_dict = {}

        if (self.check_param('age',params)):
            self.has_age=True
            self.age = parameters.CharacterParameter(
                name='age',
                minval=0.0,
                maxval=params['age_max'],
                value =0.0,
            )
            if (
                self.check_param('complex_brain',params) and
                self.check_param('complex_brain_variables',params) and
                ( 'age' in params['complex_brain_variables'] )
            ):
                brain_param_dict['age'] = self.age.get_value
        else:
            self.has_age=False
            self.age = None

        if (self.check_param('energy',params)):
            self.has_energy=True
            self.energy = parameters.Energy(
                minval                 = 0.0,
                maxval                 = params['energy_max'],
                value                  = params['energy_max'],
                energy_speed_decrement = params['energy_speed_delta'],
                energy_time_decrement  = params['energy_time_delta'],
            )
            if (
                self.check_param('complex_brain',params) and
                self.check_param('complex_brain_variables',params) and
                ( 'energy' in params['complex_brain_variables'] )
            ):
                brain_param_dict['energy'] = self.energy.get_value
        else:
            self.has_energy=False
            self.energy = None

        if (self.check_param('food_of',params)):
            self.food_of = params['food_of']

        if (self.check_param('needs_food',params)):
            self.eats = True
            self.food_source = params['food_source']
            self.prev_in_food_source = False
            self.in_food_source = False
            # TODO: Implement in complex brain
        else:
            self.eats = False
            self.food_source = None

        if (self.check_param('vision',params)):
            self.vision = True
            self.eyes = parameters.VisionObj(
                params['eye_offset'],
                params['eye_fov'],
                params['eye_dist'],
                params['eye_rays']
            )

            for obj in self.eyes.get_param("possible_objects"):
                obj_name = '_'.join(obj.split(' '))

                if (
                    self.check_param('complex_brain',params) and
                    self.check_param('complex_brain_variables',params) and
                    ( 'eyes_'+obj_name in params['complex_brain_variables'] )
                ):
                    brain_param_dict['eyes_'+obj_name] = []

                raynum = 0
                for i in range(0,self.eyes.get_param('n_rays')):
                    varname = 'eye_ray_value_'+obj_name+"_"+str(raynum)
                    brain_param_dict[varname] = (self.eyes.get_left_eye_value ,obj, i)

                    if (
                        self.check_param('complex_brain',params) and
                        self.check_param('complex_brain_variables',params) and
                        ( 'eyes_'+obj_name in params['complex_brain_variables'] )
                    ):
                        brain_param_dict['eyes_'+obj_name].append( (self.eyes.get_left_eye_value ,obj, i) )

                    raynum += 1

                for i in range(0,self.eyes.get_param('n_rays')):
                    varname = 'eye_ray_value_'+obj_name+"_"+str(raynum)

                    brain_param_dict[varname] = (self.eyes.get_right_eye_value ,obj, i)

                    if (
                        self.check_param('complex_brain',params) and
                        self.check_param('complex_brain_variables',params) and
                        ( 'eyes_'+obj_name in params['complex_brain_variables'] )
                    ):
                        brain_param_dict['eyes_'+obj_name].append( (self.eyes.get_right_eye_value ,obj, i) )

                    raynum += 1


        else:
            self.vision=False
            self.eyes=None

        if (
            self.check_param('brain',params) or
            self.check_param('complex_brain',params)
        ):
            self.interprets = True
            self.brain_param_dict = brain_param_dict
            self.brain_order = self.brain_param_dict.keys()

            self.brain_mutation_min  = params['mutation_floor']
            self.brain_mutation_max  = params['mutation_max']
            self.brain_mutation_half = params['mutation_halflife']

            if ( self.check_param('complex_brain',params) ):
                brain_struct, brain_objs, brain_dict = save_load_params.load_complex_brains(
                    params['complex_brain_input_structure'],
                    self.brain_param_dict
                )
                self.brain_order = brain_struct

                for key in brain_dict:
                    if ( isinstance(brain_dict[key], list) ):
                        for i in range(0,len(brain_dict[key])):
                            name = brain_dict[key][i]
                            if ( isinstance(name,str) and
                                ( name in brain_param_dict )
                            ):
                                brain_dict[key][i] = brain_param_dict[name]

                self.brain_param_dict = brain_dict
                self.brain = CB.ComplexBrain(
                    self.brain_order,
                    brain_objs,
                    self.brain_param_dict,
                    params['complex_brain_nn_structure'],
                    input_net_cutoffs=-2
                )
            else:
                self.brain = NN.NeuralNetwork(
                    n_inputs_init        = len(self.brain_order),
                    layer_sizes          = params['brain_layers'],
                    weights              = params['brain_weights'],
                    biases               = params['brain_biases'],
                    activation_functions = params['brain_AF'],
                )
        else:
            self.interprets = False
            self.brain = None

        if (
            self.interprets and self.has_age and self.has_energy and
            self.vision and self.eats and self.check_param('learns',params)
        ):
            self.learns = True
            self.food_orientation_reward      = params['food_source_orientation_reward']
            self.food_move_reward             = params['food_source_move_reward']
            self.self_orientation_reward      = params['self_orientation_reward']
            self.self_move_reward             = params['self_move_reward']
            self.pred_prey_orientation_reward = params['predator_prey_orientation_reward']
            self.pred_prey_move_reward        = params['predator_prey_move_reward']

            self.food_enter_reward       = params['food_source_enter_reward']
            self.feeds_reward            = params['feeds_reward']
            self.learning_max            = params['learning_max']
            self.learning_min            = params['learning_floor']
            self.learning_halflife       = params['learning_halflife']
        else:
            self.learns = False
            self.learning_max = None
            self.learning_min = None

        #TODO: implement proba
        if (self.check_param('spawns_fixed',params) or self.check_param('spawns_proba',params)):
            self.reproduces   = True
            self.spawn_last_t = 0.0
            self.spawn_adult  = False
            self.spawn_delay        = params['new_spawn_delay']
            self.spawn_energy_min   = params['spawn_energy_min']
            self.spawn_energy_delta = params['spawn_energy_delta']

            # Fixed
            self.spawn_fixed_time_min = params['spawn_time_fixed']
        else:
            self.reproduces = False
            self.spawn_last_t = None

    def __additional_equal__(self,other):
        if(
            (self.has_age != other.has_age) or
            (self.has_age and (self.age != other.age))
        ):
            return False
        if(
            (self.has_energy != other.has_energy) or
            (self.has_energy and (self.energy != other.energy))
        ):
            return False
        if(
            (self.eats != other.eats) or
            (self.eats and (self.food_source != other.food_source))
        ):
            return False
        if(
            (self.vision != other.vision) or
            (self.vision and (self.eyes != other.eyes))
        ):
            return False
        if(
            (self.interprets != other.interprets) or
            (self.interprets and (self.brain != other.brain))
        ):
            return False
        return True

    def eat(self):
        if ( not self.eats ):
            return
        self.prev_in_food_source = self.in_food_source
        self.in_food_source = True
        if ( self.has_energy ):
            self.energy.value = self.energy.get_param('max')

    def not_eat(self):
        self.prev_in_food_source = self.in_food_source
        self.in_food_source = False

    def learn_food_source(self, source_name, attempted_delta_orientation, attempted_delta_speed ):
        assert source_name in [self.food_source,self.get_name(),self.food_of]
        eyes = self.eyes

        expected_heading_angle_l = 0
        expected_heading_angle_r = 0

        l_sum = 1e-9
        r_sum = 1e-9

        l_base = eyes.left_ray_angles[0]
        r_base = eyes.right_ray_angles[0]

        for ray in range(eyes.n_rays):
            l_sum += eyes.get_left_eye_values(source_name)[ray]
            r_sum += eyes.get_right_eye_values(source_name)[ray]

            l_ori = l_base - eyes.ray_width * ray
            r_ori = 2*math.pi - (r_base - eyes.ray_width * ray)

            expected_heading_angle_l += l_ori*eyes.get_left_eye_values (source_name)[ray]
            expected_heading_angle_r += r_ori*eyes.get_right_eye_values(source_name)[ray]

        expected_heading_angle_l = expected_heading_angle_l/l_sum
        expected_heading_angle_r = expected_heading_angle_r/r_sum
        expected_heading_angle = (
            (expected_heading_angle_l * l_sum - expected_heading_angle_r * r_sum ) /
            ( l_sum + r_sum )
        )

        source_left  = eyes.left_side_has [source_name]
        source_right = eyes.right_side_has[source_name]

        l_ray = int( ( attempted_delta_orientation - l_base ) / (-eyes.ray_width) )
        r_ray = int( ( attempted_delta_orientation - r_base ) / (-eyes.ray_width) )

        delta_ori_has_source = False
        if (
            ( ( l_ray < eyes.n_rays ) and ( l_ray >= 0 ) ) or
            ( ( r_ray < eyes.n_rays ) and ( r_ray >= 0 ) )
        ):
            delta_ori_has_source = True

        speed_increase = attempted_delta_speed > 0
        speed_decrease = attempted_delta_speed < 0

        orientation_reward = 0.
        speed_reward = 0.

        if ( source_name == self.food_source ):

            orientation_reward = (
                attempted_delta_orientation - expected_heading_angle
            ) * self.food_orientation_reward

            if ( (self.get_speed() == 0) and (source_left or source_right) ):
                speed_reward = -self.food_move_reward
            elif ( delta_ori_has_source and speed_increase ):
                speed_reward = -self.food_move_reward * attempted_delta_speed
            elif ( not delta_ori_has_source and speed_decrease ):
                speed_reward = -self.food_move_reward * attempted_delta_speed
            else:
                speed_reward = self.food_move_reward * attempted_delta_speed

        elif ( source_name == self.get_name() ):
            # If right in front of character, slow down and turn

            left_heading_penalty = 0
            for ray in range(
                min(          0,eyes.left_heading_orientation_ray-1),
                max(eyes.n_rays,eyes.left_heading_orientation_ray+1)
            ):
                # Reverse vision distance formula
                dist =  ( 1. - np.sqrt(eyes.get_left_eye_values(source_name)[ray]) ) * eyes.max_dist

                mult = 1.
                if ( ray == eyes.left_heading_orientation_ray ):
                    mult = 3.

                if ( dist < 4. * self.size ):
                    left_heading_penalty += mult * eyes.get_left_eye_values(source_name)[ray]

            right_heading_penalty = 0
            for ray in range(
                min(          0,eyes.right_heading_orientation_ray-1),
                max(eyes.n_rays,eyes.right_heading_orientation_ray+1)
            ):
                # Reverse vision distance formula
                dist =  ( 1. - np.sqrt(eyes.get_right_eye_values(source_name)[ray]) ) * eyes.max_dist

                mult = 1.
                if ( ray == eyes.right_heading_orientation_ray ):
                    mult = 3.

                if ( dist < 4. * self.size ):
                    right_heading_penalty += mult * eyes.get_right_eye_values(source_name)[ray]

            if ( (left_heading_penalty > 0) or (right_heading_penalty > 0) ):

                if ( left_heading_penalty >= right_heading_penalty ):
                    orientation_reward =-self.self_orientation_reward * left_heading_penalty
                else:
                    orientation_reward = self.self_orientation_reward * right_heading_penalty

                if ( self.get_speed() > 0 ):
                    speed_reward = self.self_move_reward

        return orientation_reward, speed_reward

    def act(self,timestep):
        if ( self.learns ):
            prev_orientation = self.get_orientation()
            prev_speed       = self.get_speed()

        orientation_change_tanh, speed_change_tanh = self.interpret_input()

        orientation_change_radians = orientation_change_tanh * 2*math.pi
        delta_orientation = orientation_change_radians*timestep
        self.orientation.update(delta_orientation)

        self.speed.update(speed_change_tanh*timestep)

        if ( self.learns ):

            acceleration = ( self.get_speed() - prev_speed ) / timestep

            orientation_reward = 0
            speed_reward = 0

            # Uncomment when training
            for source in [
                #self.food_source,
                self.get_name(),
                #self.food_of
            ]:
                if ( source is None ):
                    continue

                new_orientation_reward, new_speed_reward = self.learn_food_source(
                    source,
                    orientation_change_radians,
                    acceleration
                )
                orientation_reward += new_orientation_reward
                speed_reward += new_speed_reward

            if ( isinstance(self.brain,NN.NeuralNetwork) ):
                input_list = self.get_interpret_vars()

                self.brain.backprop(
                    input_list,
                    np.array([orientation_reward,speed_reward]),
                    self.learning_rate()
                )
            # Complex brain calculates input from functions
            else:
                self.brain.backprop(
                    np.array([orientation_reward,speed_reward]),
                    self.learning_rate()
                )

    def unpack_interpret_vars(self,value):
        if ( isinstance( value, list ) ):
            return_list = []
            for v in value:
                return_list.append( self.unpack_interpret_vars( v ) )
            return return_list
        elif ( isinstance( value, tuple ) ):
            list_op = list(value)
            op = list_op.pop(0)
            tup_op = tuple(list_op)
            return [ op( *tup_op ) ]
        else:
            return [ value ]

    def get_interpret_vars(self):
        input_list = []
        for key in self.brain_order:
            var_name = key
            input_list +=  self.unpack_interpret_vars( self.brain_param_dict[key] )
        return input_list

    def interpret_input(self):
        if (not self.interprets):
            return None

        brain_results = None
        if ( isinstance(self.brain,NN.NeuralNetwork) ):
            input_list = self.get_interpret_vars()

            brain_results = self.brain.calc(input_list)
        else:
            brain_results = self.brain.calc()
        return brain_results

    def learning_rate(self):
        if ((self.learning_min is None) or (self.learning_max is None)):
            return None
        return (
            self.learning_min +
            (
                (self.learning_max-self.learning_min) /
                ( 2**(float(self.age.value)/self.learning_halflife) )
            )
        )

    def mutation_rate(self):
        if ((self.brain_mutation_min is None) or (self.brain_mutation_max is None)):
            return None
        return (
            self.brain_mutation_min +
            (
                (self.brain_mutation_max-self.brain_mutation_min) /
                ( 2**(float(self.generation)/self.brain_mutation_half) )
            )
        )

    def mutated_value(x,mutation_rate):
        return x * ( 1 + mutation_rate * 2 * ( random.random() - 0.5 ) )

    def mutate_1D_list(inp_list,mutation_rate,allow_const=False):
        new_list = []
        if ( allow_const ):
            prev_val = inp_list[0]
            all_equal = True
            for i in range(0,len(inp_list)):
                if (prev_val != inp_list[i]):
                    all_equal = False
                    break

            if all_equal:
                new_val = Prey.mutated_value(prev_val,mutation_rate)
                for j in range(0,len(inp_list)):
                    new_list.append(new_val)

                return new_val

        for i in range(0,len(inp_list)):
            new_list.append(Prey.mutated_value(inp_list[i],mutation_rate))
        return new_list

    def mutate_2D_list(weight_bias_list,mutation_rate,allow_const=False):
        new_list = []
        for i in range(0,len(weight_bias_list)):
            new_list.append(
                Prey.mutate_1D_list(weight_bias_list[i],mutation_rate,allow_const)
            )
        return new_list

    def can_spawn(self,time):
        if ( not self.reproduces ):
            return False

        self.spawn_last_t += time

        # Ensure it has existed long enough first, only has to do this once
        if (self.spawn_last_t>self.spawn_delay):
            self.spawn_adult = True

        if (not self.spawn_adult):
            return False

        # Fixed
        if (self.spawn_last_t < self.spawn_fixed_time_min):
            return False

        # Check we have enough energy
        if (self.energy is not None):
            if ( self.energy.get_value() < self.spawn_energy_min ):
                return False

        return True

    def spawn(self,x,y):
        if (not self.reproduces):
            return None
        if (self.has_energy):
            self.energy.decrease(abs(self.spawn_energy_delta))
        self.spawn_last_t = 0.0

        child = Prey(
            x,
            y,
            self.size,
            parameters.Speed(
                self.speed.get_param('max'),
                self.speed.get_param('max'),
                0.0,
            ),
            parent = self.id,
            input_parameters = self.input_parameters
        )
        child.eat()

        parent_brain = self.get_param('brain')
        mutation_rate = self.mutation_rate()
        if (mutation_rate is None):
            mutated_brain = parent_brain
        else:
            brain_inputs  = parent_brain.get_n_inputs()
            brain_layers  = parent_brain.get_layer_sizes()
            brain_weights = parent_brain.get_weights().copy()
            brain_biases  = parent_brain.get_biases().copy()
            brain_afs     = parent_brain.get_activation_functions()

            new_brain_biases = Prey.mutate_1D_list(brain_biases,mutation_rate,allow_const=True)

            new_brain_weights = []
            for layer in brain_weights:
                new_brain_weights.append(Prey.mutate_2D_list(layer,mutation_rate,allow_const=True))

            mutated_network = NN.NeuralNetwork(
                brain_inputs,
                brain_layers,
                new_brain_weights,
                new_brain_biases,
                brain_afs
            )

            if ( isinstance(self.brain,NN.NeuralNetwork) ):
                child.set_param('brain',mutated_network)
            else:
                '''
                Initial brain will have child inputs, randomly selected
                NN's for operation layer, randomly set NN.
                Need to inherit from parent.
                '''
                child.brain.set_operations( parent_brain.get_operations() )
                child.brain.set_NN( mutated_network )

        child.set_param('generation',child.get_param('generation')+1)
        return child

    def get_pickle_obj(self):
        return_list = [
            ('id'         ,self.id                ),
            ('name'       ,self.get_name()        ),
            ('consumed'   ,self.consumed          ),
            ('collision'  ,self.collision         ),
            ('x'          ,self.x                 ),
            ('y'          ,self.y                 ),
            ('speed'      ,self.get_speed()       ),
            ('orientation',self.get_orientation() ),
            ('size'       ,self.size              ),
            ('radius'     ,self.radius            ),
            ('age'        ,self.get_age()         ),
            ('energy'     ,self.get_energy()      ),
            ('food'       ,self.food_source       ),
            ('generation' ,self.generation        ),
            ('last_spawn' ,self.spawn_last_t      ),
        ]

        if ( self.eyes is None ):
            return_list.append( ('eyes',False) )
        else:
            return_list.append( ('eyes',True) )

            raynum = 0
            for rays in [
                self.eyes.get_param('left_ray_angles'),
                self.eyes.get_param('right_ray_angles')
            ]:
                for ray in rays:
                    return_list.append((
                        'eye_ray_angle_'+str(raynum),
                        ray
                    ))
                    raynum += 1

            for obj in self.eyes.get_param("possible_objects"):
                obj_name = '_'.join(obj.split(' '))
                raynum = 0
                for rays in [
                    self.eyes.get_left_eye_values(obj),
                    self.eyes.get_right_eye_values(obj)
                ]:
                    for ray in rays:
                        return_list.append((
                            'eye_ray_value_'+obj_name+"_"+str(raynum),
                            ray
                        ))
                        raynum += 1


        if ( self.brain is None ):
            return_list.append( ('brain',False) )
        else:
            return_list.append(('brain'                     ,True                                 ))
            return_list.append(('brain_inputs'              ,self.brain.get_n_inputs()            ))
            return_list.append(('brain_layer_sizes'         ,self.brain.get_layer_sizes()         ))
            return_list.append(('brain_biases'              ,self.brain.get_biases()              ))
            return_list.append(('brain_weights'             ,self.brain.get_weights()             ))
            return_list.append(('brain_activation_functions',self.brain.get_activation_functions()))
            return_list.append(('brain_field_order'         ,list(self.brain_order)               ))

        return return_list

class Predator(Prey):
    def get_name(self):
        self.name = 'predator'
        return 'predator'
