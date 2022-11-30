import random
import math
import numpy as np
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

class VisionObj:
    def __init__(
        self,
        offset_angle, # How far angular from orientation the eyes placed,
        eye_fov,
        max_sight_dist,
        n_rays,
        objs_to_see = [
            'food source',
            'prey',
        ],
    ):
        self.n_rays = n_rays
        self.max_dist = max_sight_dist
        self.offset_angle = offset_angle
        self.fov = eye_fov

        self.left  = {}
        self.right = {}
        self.possible_objects = objs_to_see
        for obj in self.possible_objects:
            self.left[obj] = np.zeros(self.n_rays)
            self.right[obj] = np.zeros(self.n_rays)

        self.ray_width = self.fov / self.n_rays
        left_min = self.offset_angle + self.fov / 2. - self.ray_width / 2.

        self.left_ray_angles = np.zeros(self.n_rays)
        for i in range(0,self.n_rays):
            self.left_ray_angles[i] = -self.ray_width*i+left_min

        self.right_ray_angles = np.zeros(self.n_rays)
        for i in range(0,self.n_rays):
            #self.right_ray_angles[i] = (-self.left_ray_angles[self.n_rays-i-1]) % (2*math.pi)
            self.right_ray_angles[i] = (-self.left_ray_angles[i]) % (2*math.pi)
        self.right_ray_angles = self.right_ray_angles[::-1]

    def eye_rays(self):
        for lr in [self.left_ray_angles,self.right_ray_angles]:
            for i in range(0,lr.shape[0]):
                yield (lr[i] + self.ray_width/2., lr[i], lr[i] - self.ray_width/2. )

    def reset_vision(self,val=0.0):
        for lr in [self.left,self.right]:
            for key in lr:
                for i in range(0,lr[key].shape[0]):
                    lr[key][i]=val

    def place_in_vision(self,obj_type,dist,left_angle,right_angle,max_dist = 10):
        vision_dist = 0
        if ( abs(dist) < 10**(-max_dist) ):
            vision_dist = max_dist
        else:
            vision_dist = min(max_dist,np.log10(self.max_dist/dist))

        for lr, init_bit_center, direction in [
            (self.left ,self.left_ray_angles[0],1),
            (self.right,self.right_ray_angles[0],-1)
        ]:
            step = self.ray_width * direction
            base = init_bit_center + direction * step / 2.

            left_bin  = ( left_angle-base) / step
            right_bin = (right_angle-base) / step

            leftmost_bin  = self.n_rays
            rightmost_bin = -1
            # Locate edges of bins
            for x_bin in [int(left_bin),int(right_bin)]:
                if (x_bin < leftmost_bin):
                    leftmost_bin = int(x_bin)
                if (x_bin > rightmost_bin):
                    rightmost_bin = int(x_bin)

            # Possible no ray directly in bin, but object crosses bin
            if (
                ( leftmost_bin < self.n_rays) and
                (rightmost_bin >= 0 )
            ):
                for this_bin in range(leftmost_bin,rightmost_bin+1):
                    if ((this_bin>=0) and (this_bin<self.n_rays)):
                        if (lr[obj_type][this_bin] < vision_dist):
                            lr[obj_type][this_bin] = vision_dist
                        

    def list_params(self):
        return self.__dict__.keys()

    def get_param(self,name):
        assert (name in self.list_params())
        return self.__dict__[name]
