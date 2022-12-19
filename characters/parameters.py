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
        out_str = str(this_dict['name']) + ":\n"
        for key in this_dict:
            if (key!='name'):
                out_str += "\t"+key+":\t"+str(this_dict[key])+"\n"
        return out_str

    def __eq__(self,other):
        return self.value == other.value

    def get_param(self,name):
        return self.__dict__[name]

    def value(self):
        return self.value

    def get_value(self):
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
                x - self.update_minus
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
    def decrease(
        self,
        x,
    ):
        self.value -= x

class VisionObj:
    def __init__(
        self,
        offset_angle, # How far angular from orientation the eyes placed,
        eye_fov,
        max_sight_dist,
        n_rays,
        objs_to_see = [
            'food source',
            #TODO: remove
            #'prey',
        ],
    ):
        self.n_rays = n_rays
        self.max_dist = max_sight_dist
        self.offset_angle = offset_angle
        self.fov = eye_fov

        self.left  = {}
        self.right = {}
        self.left_side_has = {}
        self.right_side_has = {}
        self.possible_objects = objs_to_see
        self.closest_left_bin = {}
        self.closest_right_bin = {}
        self.closest_left_bin_val = {}
        self.closest_right_bin_val = {}

        self.reset_vision()

        self.ray_width = self.fov / self.n_rays
        left_min = self.offset_angle + self.fov / 2. - self.ray_width / 2.

        self.left_heading_orientation_ray = 0
        left_heading_orientation_val = 0

        self.left_ray_angles = np.zeros(self.n_rays)
        for i in range(0,self.n_rays):
            self.left_ray_angles[i] = (-self.ray_width*i+left_min) % (2*math.pi)
            if ( abs( self.left_ray_angles[i] - math.pi ) > left_heading_orientation_val ):
                self.left_heading_orientation_ray = i
                left_heading_orientation_val = abs( self.left_ray_angles[i] - math.pi )

        self.right_heading_orientation_ray = 0
        right_heading_orientation_val = 0

        self.right_ray_angles = np.zeros(self.n_rays)
        for i in range(0,self.n_rays):
            self.right_ray_angles[i] = (-self.left_ray_angles[i]) % (2*math.pi)
            if ( abs( self.right_ray_angles[i] - math.pi ) > right_heading_orientation_val ):
                self.right_heading_orientation_ray = i
                right_heading_orientation_val = abs( self.right_ray_angles[i] - math.pi )
        self.right_ray_angles = self.right_ray_angles[::-1]
        self.bins_circle = 2*math.pi / self.fov * self.n_rays

    def __eq__(self,other):
        is_equal = (
            (self.n_rays       == other.n_rays      ) and
            (self.max_dist     == other.max_dist    ) and
            (self.offset_angle == other.offset_angle) and
            (self.fov          == other.fov         ) and
            (self.ray_width    == other.ray_width   )
        )
        if (not is_equal):
            return False
        for s, o in zip(self.left_ray_angles,other.left_ray_angles):
            if ( s != o ):
                return False
        for s, o in zip(self.right_ray_angles,other.right_ray_angles):
            if ( s != o ):
                return False
        if (self.possible_objects!=other.possible_objects):
            return False
        for obj in self.possible_objects:
            for s,o in zip(self.right[obj],other.right[obj]):
                if (s!=o):
                    return False
            for s,o in zip(self.left[obj],other.left[obj]):
                if (s!=o):
                    return False
        return True

    def eye_rays(self):
        for lr in [self.left_ray_angles,self.right_ray_angles]:
            for i in range(0,lr.shape[0]):
                yield (lr[i] + self.ray_width/2., lr[i], lr[i] - self.ray_width/2. )

    def reset_vision(self):
        for obj in self.possible_objects:
            self.left[obj] = np.zeros(self.n_rays)
            self.right[obj] = np.zeros(self.n_rays)
            self.left_side_has[obj] = False
            self.right_side_has[obj] = False
            self.closest_left_bin[obj] = 0
            self.closest_right_bin[obj] = 0
            self.closest_left_bin_val[obj] = 0.
            self.closest_right_bin_val[obj] = 0.

    def get_left_right_bins(self,orientation):
        bins = []
        for init_bit_center, direction in [
            ( self.left_ray_angles[0],1),
            (self.right_ray_angles[0],-1)
        ]:
            step = self.ray_width * direction
            base = init_bit_center + direction * step / 2.

            bins.append( ( orientation-base) / step )
        return int(bins[0]), int(bins[1])

    def place_in_vision(self,obj_type,dist,left_angle,right_angle):
        if (
            (obj_type not in self.possible_objects) or
            (dist > self.max_dist)
        ):
            return

        vision_dist = ( 1. - (dist/self.max_dist) )**2

        for lr, init_bin_center, direction, lr_str in [
            (self.left ,self.left_ray_angles[0],1,'left'),
            (self.right,self.right_ray_angles[0],-1,'right')
        ]:
            step = -self.ray_width
            base = init_bin_center - step / 2.

            left_bin  = ( left_angle - base ) / step
            right_bin = ( right_angle-left_angle ) / step + left_bin

            for this_bin_pre in range(max(0,int(left_bin)),int(right_bin)+1):
                this_bin = int(this_bin_pre % self.bins_circle)
                if (
                    ( this_bin < self.n_rays ) and
                    ( lr[obj_type][this_bin] < vision_dist )
                ):
                    lr[obj_type][this_bin] = vision_dist

                    if (
                        (lr_str == 'left') and
                        (
                            (vision_dist > self.closest_left_bin_val[obj_type]) or
                            (
                                (vision_dist == self.closest_left_bin_val[obj_type]) and
                                (
                                    abs(this_bin - self.left_heading_orientation_ray) <
                                    abs(self.closest_left_bin[obj_type] - self.left_heading_orientation_ray)
                                )
                            )
                        )
                    ):
                        self.closest_left_bin[obj_type] = this_bin
                        self.closest_left_bin_val[obj_type] = vision_dist
                    if (
                        (lr_str == 'right') and
                        (
                            (vision_dist > self.closest_right_bin_val[obj_type]) or
                            (
                                (vision_dist == self.closest_right_bin_val[obj_type]) and
                                (
                                    abs(this_bin - self.right_heading_orientation_ray) <
                                    abs(self.closest_right_bin[obj_type] - self.right_heading_orientation_ray)
                                )
                            )
                        )
                    ):
                        self.closest_right_bin[obj_type] = this_bin
                        self.closest_right_bin_val[obj_type] = vision_dist

                    if (
                        ( left_angle % (2*math.pi) < math.pi) or
                        (right_angle % (2*math.pi) < math.pi)
                    ):
                        self.left_side_has[obj_type] = True
                    if (
                        ( left_angle % (2*math.pi) > math.pi) or
                        (right_angle % (2*math.pi) > math.pi)
                    ):
                        self.right_side_has[obj_type] = True


    def list_params(self):
        return self.__dict__.keys()

    def get_param(self,name):
        assert (name in self.list_params())
        return self.__dict__[name]

    def get_left_eye_values(self,name):
        return self.left[name]

    def get_right_eye_values(self,name):
        return self.right[name]

    def get_left_eye_value(self,name,ray):
        return self.left[name][ray]

    def get_right_eye_value(self,name,ray):
        return self.right[name][ray]
