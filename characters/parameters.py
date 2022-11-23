import random
import math
import sys
import os

sys.path.append('/'.join( (os.getcwd() + '/' + __file__).split('/')[:-2] )+'/')

class CharacterParameter:
    def __init__(
        self,
        name,
        minval,
        maxval,
        value         = None,
        update_minus  = 0.0,
        update_mult   = 1.0,
    ):
        assert isinstance(minval,float)
        assert isinstance(maxval,float)
        assert isinstance(update_minus,float)
        assert isinstance(update_mult,float)
        assert maxval>= minval
        if (value is not None):
            assert isinstance(value,float)
        else:
            value = (maxval-minval)/2. + minval

        self.name  = name
        self.value = value
        self.max   = maxval
        self.min   = minval
        self.update_minus = update_minus
        self.update_mult  = update_mult

    def __str__(self):
        this_dict = self.__dict__
        out_str = str(this_dict['name']) + ":\n"#str(self.param_dict['name']) + ":\n"
        for key in this_dict:
            if (key!='name'):
                out_str += "\t"+key+":\t"+str(this_dict[key])+"\n"
        return out_str

    def get_param(self,name):
        return self.__dict__[name]

    def value(self):
        return self.value

    def set_param(self,name,value):
        assert name in self.__dict__
        self.__dict__[name] = value

    def update(self,x):
        self.value = min(
            self.max,
            max(
                self.min,
                self.value + self.update_mult * (
                    max( 0, x - self.update_minus
                       )
                )
            )
        )

class Orientation(CharacterParameter):
    def __init__(
        self,
        value         = None,
        update_minus  = 0.0,
        update_mult   = 1.0,
    ):
        assert isinstance(update_minus,float)
        assert isinstance(update_mult,float)
        if (value is not None):
            assert isinstance(value,float)
        else:
            value = 2*math.pi*random.random()
        self.name  = 'orientation'
        self.value = value % (2*math.pi)
        self.max   = 2*math.pi
        self.min   = 0.0
        self.update_minus = update_minus
        self.update_mult  = update_mult

    def update(self,x):
        self.value = (
            self.value +
            self.update_mult * (
                max( 0, x - self.update_minus )
            )
        ) % self.max

class Speed(CharacterParameter):
    def __init__(
        self,
        maxval,
        absmaxval,
        value = None,
    ):
        assert isinstance(maxval,float)
        if (value is not None):
            assert isinstance(value,float)
        else:
            value = random.random()*(maxval)/2.
        self.name  = 'speed'
        self.value = value
        self.max   = absmaxval
        self.min   = 0

    def update(self,acceleration):
        self.value = min(
            self.max,
            max(
                self.min,
                self.value + acceleration
            )
        )
    
class Energy(CharacterParameter):
    def __init__(
        self,
        minval = 0.0,
        maxval = 1.0,
        value = None,
        energy_speed_decrement = 0.0,
        energy_time_decrement = 0.0,
    ):
        assert isinstance(minval,float)
        assert isinstance(maxval,float)
        assert isinstance(energy_speed_decrement,float)
        assert isinstance(energy_time_decrement,float)
        assert maxval>= minval
        if (value is not None):
            assert isinstance(value,float)
        else:
            value = maxval

        self.name  = 'energy'
        self.value = value
        self.max   = maxval
        self.min   = minval
        self.speed_delta = abs(energy_speed_decrement)
        self.time_delta = abs(energy_time_decrement)

    def update(
        self,
        time_step,
        speed=0.0,
    ):
        self.value = min(
            self.max,
            max(
                self.min,
                self.value -
                self.speed_delta * time_step * speed -
                self.time_delta * time_step
            )
        )
