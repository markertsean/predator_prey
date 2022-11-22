import random
import math
import sys
import os

sys.path.append('/'.join( (os.getcwd() + '/' + __file__).split('/')[:-2] )+'/')

import characters.parameters as parameters

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
        parent = -1
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
        self.name = Character.name

        assert isinstance(speed,parameters.Speed)
        self.speed = speed

        if (orientation is None):
            self.orientation = parameters.Orientation(value=2*math.pi*random.random())
        elif (isinstance(orientation,float)):
            self.orientation = parameters.parameters.Orientation()
        elif (isinstance(orientation,parameters.Orientation)):
            self.orientation = parameters.Orientation
        else:
            raise ValueError("orientation must be None, float, or Orientation object")

        assert isinstance(parent,int)
        self.parent = parent

        self.__setup__()

    def __setup__(self):
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

    def get_name(self):
        return None

    def get_pos(self):
        return self.x, self.y

    def get_speed(self):
        return self.speed.value()

    def get_orientation(self):
        return self.orientation.value()
    
    def percieve(self):
        pass

class Prey(Character):
    def get_name(self):
        return 'prey'

    def __setup__(self):
        self.collision=True
