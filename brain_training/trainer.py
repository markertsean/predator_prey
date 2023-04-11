import pickle as pkl
from datetime import datetime
import numpy as np
import math
import random
import sys
import os
import ast

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
    if (halflife is None):
        return lr
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
    elif ( ('[' in val) or (']' in val) ):
        print(val)
        if ( '[' in val[1:-1] ):
            return ast.literal_eval(val)
        else:
            return [interpret_conf(x) for x in val.strip('[]').split(',')]
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

def iterate_exp_array(inp_array,n,ind=-1):
    inp_array[ind] += 1
    if ( inp_array[ind] >= n ):
        pass
        # max 8, [ 9 ] -> [ 0, 1 ]
        # [ 0, 9 ] -> [ 1, 2 ]
        # [ 8, 9 ] -> [ 0, 1, 2 ]
        # [ 0, 8, 9 ] -> [ 1, 2, 3 ]
        if ( abs(ind) == inp_array.shape[0] ):
            inp_array = np.zeros(inp_array.shape[0]+1)
            for i in range(0,inp_array.shape[0]):
                inp_array[i] = i
        else:
            inp_array = iterate_exp_array( inp_array, n, ind-1 )
            inp_array[ind] = inp_array[ind-1] + 1
    return inp_array

# Inefficient but oh well
def test_vision(inp_eyes,inp_nn,inp_params):
    global training_dict
    method = inp_params['training_method']
    n_inputs = 2*inp_eyes.n_rays

    inputs = np.zeros( n_inputs )
    pred_ori, pred_speed = inp_nn.calc( inputs )
    print("Pred ori change: {:+05.1f} deg\tPred speed change: {:+05.3f}\tAngles: {}".format(
        pred_ori*180/math.pi,
        pred_speed,
        []
    ))

    exp_array = np.zeros( 1 )
    max_iter = 1e3
    c = 0
    while c < max_iter:
        c+=1

        inputs = np.zeros( n_inputs )
        angles = []
        bad = False
        for item in exp_array:
            ind = round(item)
            if ( ind < n_inputs ):
                inputs[ind] = inp_params['vision_test_val']
                if ( ind < inp_eyes.n_rays ):
                    angles.append( round(180/math.pi*inp_eyes.left_ray_angles[ind]) )
                else:
                    angles.append( round(180/math.pi*inp_eyes.right_ray_angles[ind-n_inputs]) )
            else:
                bad = True
        if ( bad ):
            continue

        if ( inp_params['vision_test'] ):
            pred_ori, pred_speed = inp_nn.calc( inputs )
            print("Pred ori change: {:+05.1f} deg\tPred speed change: {:+05.3f}\tAngles: {}".format(
                pred_ori*180,
                pred_speed,
                angles
            ))


        elif( inp_params['dry_run'] ):
            inp_eyes.reset_vision()
            inp_eyes.left['dummy' ] = inputs[:inp_eyes.n_rays]
            inp_eyes.right['dummy'] = inputs[inp_eyes.n_rays:2*inp_eyes.n_rays]
            neg_ori_pred, neg_s_pred = training_dict[method](
                0.0,
                0.0,
                inp_eyes,
                inp_params
            )
            print("Calc ori change: {:+05.1f} deg\tPred speed change: {:+05.3f}\tAngles: {}".format(
                -neg_ori_pred*180/math.pi,
                -neg_s_pred,
                angles
            ))

        exp_array = iterate_exp_array( exp_array, n_inputs )
        if ( exp_array.shape[0] > inp_params['vision_test_objs'] ):
            break

# For food, want to turn towards where the most or closest food is
def train_food(attempted_delta_orientation, attempted_delta_speed, inp_eyes, inp_params):
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

    avg_val = 0.0
    n_rays_w_val = 1e-9

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

        if ( l_val > 1e-9 ):
            n_rays_w_val += 1
            avg_val += l_val
        if ( r_val > 1e-9 ):
            n_rays_w_val += 1
            avg_val += r_val

        if ( inp_eyes.get_left_eye_values(source_name)[ray] >= closest_ray_l_val ):
            closest_ray_l_val = inp_eyes.get_left_eye_values(source_name)[ray]
            closest_ray_l_i = ray

        if ( inp_eyes.get_right_eye_values(source_name)[ray] > closest_ray_r_val ):
            closest_ray_r_val = inp_eyes.get_right_eye_values(source_name)[ray]
            closest_ray_r_i = ray

    avg_val = avg_val / n_rays_w_val

    expected_heading_angle_l = expected_heading_angle_l/l_sum
    expected_heading_angle_r = expected_heading_angle_r/r_sum
    expected_heading_angle = (
        ( (expected_heading_angle_l * l_sum) + (expected_heading_angle_r * r_sum) ) /
        ( l_sum + r_sum )
    )

    closest_ray_angle = attempted_delta_orientation
    closest_ray_angle = inp_eyes.left_ray_angles[closest_ray_l_i]
    closest_ray_val   = closest_ray_l_val
    if ( closest_ray_r_val > closest_ray_l_val ):
        closest_ray_angle = -(2*math.pi - inp_eyes.right_ray_angles[closest_ray_r_i])
        closest_ray_val   = closest_ray_r_val

    # If much closer to one source
    #######################
    if ( True):#closest_ray_val > 1.3 * avg_val ):
        if (closest_ray_angle > math.pi):
            closest_ray_angle = 2*math.pi - closest_ray_angle

        #print("Old heading ray:",expected_heading_angle * 180/math.pi)
        expected_heading_angle = closest_ray_angle
        #print("Using closest ray:",expected_heading_angle * 180/math.pi)

    speed_increase = attempted_delta_speed > 0
    speed_decrease = attempted_delta_speed < 0

    food_orientation_reward = 1.0
    food_move_reward = 1.0

    orientation_reward = 0.
    speed_reward = 0.

    # If pointing the right way, reward correctly (neg for desired increase, pos for decrease)
    if ( (l_sum < 1.e-8) and (r_sum < 1.e-8) ):
        #print("No food")
        speed_reward = food_move_reward * ( attempted_delta_speed - 1 )
        expected_heading_angle = inp_eyes.ray_width
    elif (
        abs(expected_heading_angle) < 2*inp_eyes.ray_width
    ):
        #print("Food in heading")
        speed_reward = food_move_reward * ( attempted_delta_speed - 1 )
    else:
        #print("Food not in heading")
        speed_reward = food_move_reward * ( attempted_delta_speed + 1 )

    orientation_reward = (
        attempted_delta_orientation - expected_heading_angle
    ) * food_orientation_reward

    #print("Expected:",expected_heading_angle * 180/math.pi)

    return orientation_reward, speed_reward

# For self, want to turn/slow when we are heading right at it closest food is
def train_self(attempted_delta_orientation, attempted_delta_speed, inp_eyes, inp_params):
    source_name = 'dummy'
    closeness = 1.0

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

        orientation_reward_score = inp_params['train_ori_mod']
        speed_reward_score = inp_params['train_speed_mod']

        main_ray = inp_eyes.left_ray_angles[leftmost_ray]
        if ( right_heading_penalty >= left_heading_penalty ):
            main_ray = inp_eyes.right_ray_angles[rightmost_ray] - 2 * math.pi

        turn_ray = math.pi/2
        inp_eyes.left_ray_angles[
            max(
                inp_eyes.left_heading_orientation_ray-1,
                0
            )
        ]
        if ( left_heading_penalty >= right_heading_penalty ):
            turn_ray =-math.pi/2
            inp_eyes.right_ray_angles[
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
        return attempted_delta_orientation, attempted_delta_speed

# For prey, need to recognize predator and turn away sharply
def train_my_predator(attempted_delta_orientation, attempted_delta_speed, inp_eyes, inp_params):
    source_name = 'dummy'
    closeness = 0.00
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

    x_sum = 0.
    x_n   = 1.e-1

    y_sum = 0.
    y_n   = 1.e-1
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

        x_sum += l_val * np.cos(inp_eyes.left_ray_angles[ray]) + r_val * np.cos(inp_eyes.right_ray_angles[ray])
        x_n   += l_val + r_val

        y_sum += l_val * np.sin(inp_eyes.left_ray_angles[ray]) + r_val * np.sin(inp_eyes.right_ray_angles[ray])
        y_n   += l_val + r_val

        if (
            (inp_eyes.get_left_eye_values(source_name)[ray] >= left_dist) and
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

    if ( (left_dist > 0) or (right_dist > 0) ):

        orientation_reward_score = 1.0
        speed_reward_score = 1.0

        #TODO: Should try to implement this, when possible, to shoot gaps
        heading_avg /= val_sum
        opposite_heading = ( heading_avg + math.pi )
        opposite_heading = opposite_heading % ( 2 * math.pi )
        if ( opposite_heading > math.pi ):
            opposite_heading = opposite_heading - 2 * math.pi

        x_avg = x_sum / x_n
        y_avg = y_sum / y_n

        avg_dist  = np.sqrt( x_avg**2 + y_avg**2 )
        avg_angle = math.atan2( y_avg, x_avg )

        #print("\tAvg angle:",avg_angle*180/math.pi)
        turn_ray = avg_angle + math.pi#heading_avg + math.pi / 2
        if ( turn_ray > math.pi ):
            turn_ray = turn_ray - 2 * math.pi
        #if ( heading_avg < 0 ):
        #    turn_ray = heading_avg - math.pi / 2
        ## Turn away from closest predators
        if ( left_dist >= right_dist ):
            speed_dist = left_dist
        #
        #    main_ray = inp_eyes.left_ray_angles[left_ray]
        #
        #    # Default, turn right
        #    turn_ray = main_ray - math.pi / 2
        #
        #    # If much is right, turn left
        #    if ( right_sum > left_sum ):
        #        turn_ray = main_ray + math.pi / 2
        else:
            speed_dist = right_dist
        #
        #    main_ray = inp_eyes.right_ray_angles[right_ray] - 2 * math.pi
        #
        #    # Default, turn left
        #    turn_ray = main_ray + math.pi / 2
        #
        #    # Much is left, turn right
        #    if ( left_sum > right_sum ):
        #        turn_ray = main_ray - math.pi / 2

        turn_ray = max(minturn,min(maxturn,turn_ray))

        orientation_reward = orientation_reward_score * (
            attempted_delta_orientation - turn_ray
        )

        # Want speed to decrease if near our heading, speed up if to the side

        # Not actual distance, but vision distance metric (bigger is closer)
        # 0 deg expect 1, +-90 deg expect -1, >90 expect -1
        ang_factor = math.cos(2*turn_ray)
        if ( abs(turn_ray) > math.pi/2 ):
            ang_factor = -1.
        expected_speed_delta = speed_dist * ang_factor
        speed_reward = speed_reward_score * ( attempted_delta_speed - expected_speed_delta )

        #print("\tAng factor :",ang_factor)
        #print("\tHeading Avg:",heading_avg * 180/math.pi)
        #print("\tTurn ray   :",turn_ray * 180/math.pi)

        return orientation_reward, speed_reward
    else:
        return attempted_delta_orientation, attempted_delta_speed

def set_vision( inp_eyes, param_dict ):
    inp_eyes.reset_vision()

    size = param_dict['train_obj_size']
    for i in range( 0, param_dict['n_train_obj'] ):
        dist = 1.5 * random.random() + size
        angle_l = random.random() * 2 * math.pi
        angle_r = 2 * math.atan( size / ( 2 * dist ) ) + angle_l

        inp_eyes.place_in_vision('dummy',dist,angle_l,angle_r)

def train( inp_eyes, inp_nn, param_dict ):
    iter_max = param_dict['epochs']

    method = param_dict['training_method']
    global training_dict
    assert method in training_dict

    print(inp_nn)

    zero_inputs = 0.

    for i in range( 0, iter_max ):

        # Need the input in the eyes, can pull from them for calc
        set_vision( inp_eyes, param_dict )
        inputs = []
        no_vals = True
        for side in [inp_eyes.left,inp_eyes.right]:
            for val in side['dummy']:
                if (val > 0):
                    no_vals = False
                inputs.append(val)
        if (no_vals):
            zero_inputs += 1

        orientation_change_tanh, speed_change_tanh = inp_nn.calc( inputs )
        orientation_change_radians = math.pi * orientation_change_tanh

        #print(30*"=")
        ##print([ str(round(x*180/math.pi)) for x in inp_eyes.left_ray_angles ])
        ##print([ str(round(x*180/math.pi)) for x in inp_eyes.right_ray_angles])
        #print("Ray width:",inp_eyes.ray_width*180/math.pi)
        ##print(inp_eyes.left['dummy'])
        ##print(inp_eyes.right['dummy'])
        #for i in range(0,len(inp_eyes.left_ray_angles)):
        #    if ( inp_eyes.left['dummy'][i] > 0 ):
        #        print(round(inp_eyes.left_ray_angles[i]*180/math.pi),inp_eyes.left['dummy'][i],end="\t")
        #for i in range(0,len(inp_eyes.right_ray_angles)):
        #    if ( inp_eyes.right['dummy'][i] > 0 ):
        #        print(round(inp_eyes.right_ray_angles[i]*180/math.pi),inp_eyes.right['dummy'][i],end="\t")
        #print()
        #print("Ori att:",orientation_change_tanh)
        #print("Ori ang:",orientation_change_radians*180/math.pi)

        o_err_rad, s_err = training_dict[method](
            orientation_change_radians,
            speed_change_tanh,
            inp_eyes,
            param_dict
        )
        o_err = o_err_rad / math.pi

        #print("Ori err ang:",o_err_rad*180/math.pi)
        #print("Ori err tan:",o_err)
        #print("Speed att:",speed_change_tanh)
        #print("Speed err:",s_err)
        '''
        Want pred value to decrease - positive error
        Want pred value to increase - negative error
        '''
        inp_nn.backprop(
            inputs,
            [o_err,s_err],
            learning_rate(i,param_dict['NN_learning_halflife'],param_dict['NN_learning_rate'])
        )
        #if ( ( i % (param_dict['epochs']/100) == 0 ) and (i>0) ):
        if ( ( i % (param_dict['epochs']/10) == 0 ) and (i>0) ):
            print("Trained epoch: {:09}".format(i))
            #print(inp_nn)

    print("{} epochs, {:05.1f}% were zero inputs".format(iter_max,100*zero_inputs/iter_max))

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
    # Only need to train if we need the net
    if ( in_dict['save'] or in_dict['vision_test'] ):
        train(eyes,base_NN,in_dict)

    if ( in_dict['save'] ):
        save_nn( eyes, base_NN, in_dict )

    if ( in_dict['vision_test'] or in_dict['dry_run'] ):
        test_vision(eyes,base_NN,in_dict)

training_dict = {
    'food':train_food,
    'self':train_self,
    'prey':train_self,
    'predator':train_my_predator,
}

if __name__ == '__main__':
    main()
