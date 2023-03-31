import pickle as pkl
from datetime import datetime
import numpy as np
import math
import random
import sys
import os

HOMEDIR = '/'.join( __file__.split('/')[:-2] )+'/'
sys.path.append(HOMEDIR)

import perceptron.neural_net as NN
import perceptron.complex_brain as CB
from perceptron import activation
import characters.parameters

AF_DICT = {
    'identity':activation.identity,
    'logistic':activation.logistic,
    'tanh':activation.tanh,
    'relu':activation.relu,
}

AF_DICT_T = {}
for key in AF_DICT:
    AF_DICT_T[AF_DICT[key]] = key

def learning_rate(epoch,halflife,lr):
    return lr * ( 2.**(-1.0*epoch/halflife) )

def interpret_conf(val):
    if (val == 'None'):
        return None
    elif (val in AF_DICT):
        return AF_DICT[val]
    elif (val.upper()=="TRUE"):
        return True
    elif (val.upper()=="FALSE"):
        return False
    elif (val[0].isalpha()):
        return val
    elif ('[' in val):
        vals = val.strip('[]').split(',')
        out_list = []
        for v in vals:
            out_list.append(interpret_conf(v))
        return out_list
    elif ('.' in val):
        return float(val)
    elif ('e' in val):
        return float(val)
    elif (val[0].isdigit()):
        return int(val)
    else:
        assert False, val + " is of unknown type!"

def read_conf():
    global HOMEDIR
    out_dict = {}
    with open(HOMEDIR+'/brain_training/trainer.conf','r') as f:
        for line in f:
            l = line.split(':')
            key = l[0].strip()
            val = l[1].strip()
            out_dict[key] = interpret_conf(val)

    return out_dict

def save_nn(inp_eyes,inp_nn,param_dict):
    global HOMEDIR
    global AF_DICT_T
    out_path = HOMEDIR+'data/brains/gen_brain_off_{:4.1f}_fov_{:4.1f}_rays_{}_layers_{}_af_{}/'.format(
        inp_eyes.offset_angle * 180/math.pi,
        inp_eyes.fov * 180/math.pi,
        inp_eyes.n_rays,
        '_'.join([ str(x) for x in inp_nn.get_layer_sizes() ]),
        '_'.join([ str(AF_DICT_T[x]) for x in param_dict['NN_layer_afs'] ])
    )
    out_fn = param_dict['out_file_name']
    if ( param_dict['append_date'] ):
        out_fn += "_"+datetime.today().strftime('%Y.%m.%d.%H.%M.%S')
    out_full = out_path + out_fn + '.pkl'

    brain_var_list = []
    for key in param_dict:
        if ("brain_var_" in key):
            brain_var_list.append(key)

    var_order_list = []
    for i in range( 2 * inp_eyes.n_rays ):
        var_list = []
        for key in sorted( brain_var_list ):
            if ( param_dict[key] == 'i' ):
                var_list.append(i)
            else:
                var_list.append( param_dict[key] )
        var_order_list.append( tuple(var_list) )

    brain_order_list = []
    for tup in var_order_list:
        brain_order_list.append(
            param_dict['brain_order'].format( *tup )
        )

    n  = inp_nn.get_n_inputs()
    l  = inp_nn.get_layer_sizes()
    w  = inp_nn.get_weights()
    b  = inp_nn.get_biases()
    af = inp_nn.get_activation_functions()
    od = brain_order_list
    # Pass list for backwards compatibility
    brain_output = [ (n,l,w,b,af,od) ]

    os.makedirs(out_path,exist_ok=True)
    with open(out_full,'wb') as f:
        pkl.dump(brain_output,f)
    print("Wrote ",out_full)

# For food, want to turn towards where the most or closest food is
def train_food(attempted_delta_orientation, attempted_delta_speed, inp_eyes):
    source_name = 'dummy'

    expected_heading_angle_l = 0
    expected_heading_angle_r = 0

    l_sum = 1e-9
    r_sum = 1e-9

    l_base = inp_eyes.left_ray_angles[0]
    r_base = inp_eyes.right_ray_angles[0]

    closest_ray_l_i = 0
    closest_ray_r_i = 0

    closest_ray_l_val = 0.
    closest_ray_r_val = 0.

    for ray in range(inp_eyes.n_rays):
        l_val = inp_eyes.get_left_eye_values(source_name)[ray]
        r_val = inp_eyes.get_right_eye_values(source_name)[ray]

        l_sum += l_val
        r_sum += r_val

        ang = inp_eyes.left_ray_angles[ray]
        if ( ang > math.pi ):
            ang = -(2*math.pi - ang)
        expected_heading_angle_l += l_val*ang

        ang = inp_eyes.right_ray_angles[ray]
        if ( ang > math.pi ):
            ang = -(2*math.pi - ang)
        expected_heading_angle_r += r_val*ang

        if ( inp_eyes.get_left_eye_values(source_name)[ray] > closest_ray_l_val ):
            closest_ray_l_val = inp_eyes.get_left_eye_values(source_name)[ray]
            closest_ray_l_i = ray

        if ( inp_eyes.get_right_eye_values(source_name)[ray] > closest_ray_r_val ):
            closest_ray_r_val = inp_eyes.get_right_eye_values(source_name)[ray]
            closest_ray_r_i = ray

    expected_heading_angle_l = expected_heading_angle_l/l_sum
    expected_heading_angle_r = expected_heading_angle_r/r_sum
    expected_heading_angle = (
        ( (expected_heading_angle_l * l_sum) + (expected_heading_angle_r * r_sum) ) /
        ( l_sum + r_sum )
    )

    source_left  = inp_eyes.left_side_has [source_name]
    source_right = inp_eyes.right_side_has[source_name]

    l_ray = int( ( attempted_delta_orientation - l_base ) / (-inp_eyes.ray_width) )
    r_ray = int( ( attempted_delta_orientation - r_base ) / (-inp_eyes.ray_width) )

    delta_ori_has_source = False
    if (
        ( ( l_ray < inp_eyes.n_rays ) and ( l_ray >= 0 ) ) or
        ( ( r_ray < inp_eyes.n_rays ) and ( r_ray >= 0 ) )
    ):
        delta_ori_has_source = True

    closest_ray_angle = attempted_delta_orientation
    closest_ray_angle = inp_eyes.left_ray_angles[closest_ray_l_i]
    if ( closest_ray_r_val > closest_ray_l_val ):
        closest_ray_angle = inp_eyes.right_ray_angles[closest_ray_r_i]

    if (closest_ray_angle > math.pi):
        closest_ray_angle = 2*math.pi - closest_ray_angle


    speed_increase = attempted_delta_speed > 0
    speed_decrease = attempted_delta_speed < 0

    food_orientation_reward = 1.0
    food_move_reward = 1.0

    orientation_reward = 0.
    speed_reward = 0.

    orientation_reward = (
        attempted_delta_orientation - expected_heading_angle
    ) * food_orientation_reward

    # If pointing the right way, reward correctly (neg for desired increase, pos for decrease)
    if ( abs( attempted_delta_orientation - expected_heading_angle ) < inp_eyes.ray_width ):
        speed_reward = -food_move_reward * attempted_delta_speed
    else:
        speed_reward = food_move_reward * attempted_delta_speed * inp_eyes.n_rays

    return orientation_reward, speed_reward

# For self, want to turn/slow when we are heading right at it closest food is
def train_self(attempted_delta_orientation, attempted_delta_speed, inp_eyes):
    source_name = 'dummy'
    closeness = 0.15

    # If right in front of character, slow down and turn

    left_heading_penalty = 0
    leftmost_dist = 1e3
    leftmost_ray = inp_eyes.left_heading_orientation_ray
    for ray in range(
        max(              0,inp_eyes.left_heading_orientation_ray-1),
        min(inp_eyes.n_rays,inp_eyes.left_heading_orientation_ray+2)
    ):
        # Reverse vision distance formula
        dist =  ( 1. - np.sqrt(inp_eyes.get_left_eye_values(source_name)[ray]) )

        if ( dist < closeness ):
            left_heading_penalty += inp_eyes.get_left_eye_values(source_name)[ray]

            if (dist < leftmost_dist):
                leftmost_dist = dist
                leftmost_ray = ray

    right_heading_penalty = 0
    rightmost_dist = 1e3
    rightmost_ray = inp_eyes.right_heading_orientation_ray
    for ray in range(
        max(              0,inp_eyes.right_heading_orientation_ray-1),
        min(inp_eyes.n_rays,inp_eyes.right_heading_orientation_ray+2)
    ):
        # Reverse vision distance formula
        dist =  ( 1. - np.sqrt(inp_eyes.get_right_eye_values(source_name)[ray]) )

        if ( dist < closeness ):
            right_heading_penalty += inp_eyes.get_right_eye_values(source_name)[ray]

            if (dist < rightmost_dist):
                righttmost_dist = dist
                rightmost_ray = ray

    if ( (left_heading_penalty > 0) or (right_heading_penalty > 0) ):

        orientation_reward_score = 1.0
        speed_reward_score = 1.0

        main_ray = inp_eyes.left_ray_angles[leftmost_ray]
        if ( right_heading_penalty >= left_heading_penalty ):
            main_ray = inp_eyes.right_ray_angles[rightmost_ray] - 2 * math.pi

        turn_ray = inp_eyes.left_ray_angles[
            max(
                inp_eyes.left_heading_orientation_ray-1,
                0
            )
        ]
        if ( left_heading_penalty >= right_heading_penalty ):
            turn_ray = inp_eyes.right_ray_angles[
                min(
                    inp_eyes.right_heading_orientation_ray+1,
                    inp_eyes.n_rays-1
                )
            ] - 2 * math.pi

        orientation_reward = orientation_reward_score * (
            attempted_delta_orientation - turn_ray
        )

        # Always should be -1
        speed_reward = speed_reward_score * ( attempted_delta_speed + 1 )

        return orientation_reward, speed_reward
    else:
        return 0., -1e-2,

# For prey, need to recognize predator and turn away sharply
def train_my_predator(attempted_delta_orientation, attempted_delta_speed, inp_eyes):
    source_name = 'dummy'
    closeness = 0.15
    maxturn = math.pi * 1.0
    minturn =-math.pi * 1.0
    '''
    Turn away from side with closest predator, 180 degree
    '''
    left_dist = 0.
    left_ray = 0
    left_sum = 0.

    right_dist = 0.
    right_ray = 0
    right_sum = 0.

    val_sum = 1.e-9
    heading_avg = 0.
    for ray in range( inp_eyes.n_rays ):
        l_val = inp_eyes.get_left_eye_values(source_name)[ray]
        r_val = inp_eyes.get_right_eye_values(source_name)[ray]

        left_sum += l_val
        right_sum += r_val

        val_sum += l_val + r_val
        heading_avg += (
            l_val * inp_eyes.left_ray_angles [ray] +
            r_val * ( inp_eyes.right_ray_angles[ray] - 2 * math.pi )
        )

        if (
            (inp_eyes.get_left_eye_values(source_name)[ray] > left_dist) and
            (inp_eyes.get_left_eye_values(source_name)[ray] > closeness)
        ):
            left_dist = inp_eyes.get_left_eye_values(source_name)[ray]
            left_ray = ray

        if (
            (inp_eyes.get_right_eye_values(source_name)[ray] > right_dist) and
            (inp_eyes.get_right_eye_values(source_name)[ray] > closeness )
        ):
            right_dist = inp_eyes.get_right_eye_values(source_name)[ray]
            right_ray = ray

    #TODO: Should try to implement this, when possible, to shoot gaps
    heading_avg /= val_sum

    if ( (left_dist > 0) or (right_dist > 0) ):

        orientation_reward_score = 1.0
        speed_reward_score = 1.0

        # Turn away from predators
        if ( left_dist >= right_dist ):
            speed_dist = left_dist

            main_ray = inp_eyes.left_ray_angles[left_ray]

            # Default, turn right
            turn_ray = main_ray - math.pi / 2

            # If much is right, turn left
            if ( right_sum > left_sum ):
                turn_ray = main_ray + math.pi / 2
        else:
            speed_dist = right_dist

            main_ray = inp_eyes.right_ray_angles[right_ray] - 2 * math.pi

            # Default, turn left
            turn_ray = main_ray + math.pi / 2

            # Much is left, turn right
            if ( left_sum > right_sum ):
                turn_ray = main_ray - math.pi / 2

        turn_ray = max(minturn,min(maxturn,turn_ray))

        orientation_reward = orientation_reward_score * (
            attempted_delta_orientation - turn_ray
        )

        # Want speed to decrease if near our heading, speed up if to the side

        # Not actual distance, but vision distance metric (bigger is closer)
        speed_dist = speed_dist

        ang_factor = math.cos(2*main_ray)

        expected_speed_delta = -speed_dist * ang_factor

        speed_reward = speed_reward_score * ( attempted_delta_speed - expected_speed_delta )

        return orientation_reward, speed_reward
    else:
        return 0., -1e-2,

def train( inp_eyes, inp_nn, param_dict ):
    iter_max = param_dict['epochs']

    method = param_dict['training_method']
    training_dict = {
        'food':train_food,
        'self':train_self,
        'prey':train_self,
        'predator':train_my_predator,
    }
    assert method in training_dict

    print(inp_nn)
    
    for i in range( iter_max ):
        inputs = param_dict['NN_min_val'] + (
                param_dict['NN_max_val'] - param_dict['NN_min_val']
            ) * np.random.rand( 2 * inp_eyes.n_rays )
        for j in range(inputs.shape[0]):
            if (random.random() < param_dict['zero_proba']):
                inputs[j] = 0.

        # Need the input in the eyes
        inp_eyes.reset_vision()
        for j in range( param_dict['eye_n_rays'] ):
            inp_eyes.left ['dummy'][j] = inputs[j]
            inp_eyes.right['dummy'][j] = inputs[j+param_dict['eye_n_rays']]

        if ( np.sum( inp_eyes.left['dummy'] > 0 ) ):
             inp_eyes.left_side_has['dummy'] = True
        if ( np.sum( inp_eyes.right['dummy'] > 0 ) ):
             inp_eyes.right_side_has['dummy'] = True

        orientation_change_tanh, speed_change_tanh = inp_nn.calc( inputs )
        orientation_change_radians = math.pi * orientation_change_tanh

        o_err_rad, s_err = training_dict[method](
            orientation_change_radians,
            speed_change_tanh,
            inp_eyes
        )
        o_err = o_err_rad / math.pi

        '''
        Want pred value to decrease - positive error
        Want pred value to increase - negative error
        '''
        inp_nn.backprop(
            inputs,
            [o_err,s_err],
            learning_rate(i,param_dict['NN_learning_halflife'],param_dict['NN_learning_rate'])
        )
        if ( ( i % (param_dict['epochs']/100) == 0 ) and (i>0) ):
            print("Trained epoch: {:09}".format(i))
            #print(inp_nn)

    print(inp_nn)

def main():
    in_dict = read_conf()

    print("Running trainer with params:")
    for key in in_dict:
        print("{:40s}\t{}".format(key,in_dict[key]))
    print()

    eyes = characters.parameters.VisionObj(
        in_dict['eye_offset'] * math.pi / 180,
        in_dict['eye_FOV']    * math.pi / 180,
        1.0,
        in_dict['eye_n_rays'],
        ['dummy'],
    )

    base_NN = NN.NeuralNetwork(
        in_dict['eye_n_rays'] * 2,
        in_dict['NN_layers'],
        in_dict['NN_weights'],
        in_dict['NN_biases'],
        in_dict['NN_layer_afs'],
    )
    train(eyes,base_NN,in_dict)

    save_nn( eyes, base_NN, in_dict )

if __name__ == '__main__':
    main()
