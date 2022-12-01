import random
import math
import sys
import os

sys.path.append('/'.join( __file__.split('/')[:-2] )+'/')

import characters.parameters as parameters
import perceptron.neural_net as NN

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
        pass

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
        self.name = 'food source'
        self.consumed=False

    def get_name(self):
        return self.name

class Prey(Character):
    def get_name(self):
        return 'prey'

    def check_param(self,key,param):
        return (key in param) and param[key]

    def __setup__(self,params):
        self.collision=True
        self.consumed=True

        brain_param_dict = {}

        if (self.check_param('prey_age',params)):
            self.age = parameters.CharacterParameter(
                name='age',
                minval=0.0,
                maxval=params['prey_age_max'],
                value =0.0,
            )
            brain_param_dict['age'] = self.age.get_value
        else:
            self.age = None

        if (self.check_param('prey_energy',params)):
            self.energy = parameters.Energy(
                minval                 = 0.0,
                maxval                 = params['prey_energy_max'],
                value                  = params['prey_energy_max'],
                energy_speed_decrement = params['prey_energy_speed_delta'],
                energy_time_decrement  = params['prey_energy_time_delta'],
            )
            brain_param_dict['energy'] = self.energy.get_value
        else:
            self.energy = None

        if (self.check_param('prey_needs_food',params)):
            self.eats = True
            self.food_source = 'food source'
        else:
            self.eats = False

        if (self.check_param('prey_vision',params)):
            self.vision = True
            self.eyes = parameters.VisionObj(
                params['prey_eye_offset'],
                params['prey_eye_fov'],
                params['prey_eye_dist'],
                params['prey_eye_rays']
            )

            for obj in self.eyes.get_param("possible_objects"):
                obj_name = '_'.join(obj.split(' '))

                for i in range(0,self.eyes.get_param('n_rays')):
                    brain_param_dict[ 'left_eye_'+obj_name+"_"+str(i)] = (self.eyes.get_left_eye_value ,obj, i)
                
                for i in range(0,self.eyes.get_param('n_rays')):
                    brain_param_dict['right_eye_'+obj_name+"_"+str(i)] = (self.eyes.get_right_eye_value,obj, i)
        else:
            self.vision=False

        if (self.check_param('prey_brain',params)):
            self.interprets = True
            self.brain_param_dict = brain_param_dict
            self.brain_order = self.brain_param_dict.keys()

            self.brain = NN.NeuralNetwork(
                n_inputs_init        = len(self.brain_order),
                layer_sizes          = params['prey_brain_layers'],
                weights              = params['prey_brain_weights'],
                biases               = params['prey_brain_biases'],
                activation_functions = params['prey_brain_AF'],
            )
        else:
            self.interprets = False

    def eat(self):
        self.energy.value = self.energy.get_param('max')

    def spawn(self,time):
        return False
        #if (not self.reproduces):
        #    return False

    def act(self,timestep):
        orientation_change, speed_change = self.interpret_input()
        #TODO: need to have resistance, consider momentum
        self.orientation.update(orientation_change*timestep)
        self.speed.update(speed_change*timestep)

    def interpret_input(self):
        if (not self.interprets):
            return None

        input_list = []
        for key in self.brain_order:
            var_name = key
            value = 0
            if ( isinstance(self.brain_param_dict[key], tuple ) ):
                func, name, i = self.brain_param_dict[key]
                value = func(name,i)
            else:
                value = self.brain_param_dict[key]()

            input_list.append( value )

        return self.brain.calc(input_list)
