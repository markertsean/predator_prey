from datetime import datetime
import multiprocessing as mp
from pathos.pools import ProcessPool,ThreadPool,ParallelPool
from multiprocess.dummy import Pool
import dill
import math
import numpy as np
import random
import pickle as pkl
import sys
import os

__project_path__='/'.join( __file__.split('/')[:-2] )+'/'
sys.path.append(__project_path__)

import characters.characters as characters

def overlap( x1, y1, r1, x2, y2, r2 ):
    dist = np.sqrt( (x2-x1)**2 + (y2-y1)**2 )
    if ( dist>r1+r2 ):
        return 0
    if ( dist==r1+r2 ):
        return 1
    return 2

def distance_two_points(x1,y1,x2,y2):
    return np.sqrt( (x1-x2)**2 + (y1-y2)**2 )

def distance_two_circles(x1,y1,r1,x2,y2,r2):
    return max(distance_two_points(x1,y1,x2,y2)-r1-r2,0)

def relative_angle_between_characters(char_1,char_2):
    return math.atan2(
        (char_2.get_param('y')-char_1.get_param('y')),
        (char_2.get_param('x')-char_1.get_param('x'))
    )

class SimulationBox:
    def __init__(
        self,
        box_size,
        cell_size,
        n_steps,
        time_step,
        max_speed,
        snapshot_step,
        max_characters,
        seed,
        n_jobs = 1,
        parallel_char_min = None,
        output_path = __project_path__+"data/",
        kill_early = True
    ):
        assert isinstance(box_size,float) or isinstance(box_size,int)
        self.length = float(box_size)

        assert isinstance(cell_size,float) or isinstance(cell_size,int)
        assert cell_size<=self.length
        self.n_cells = round(self.length/cell_size)
        self.cell_length = float(self.length/self.n_cells)

        assert isinstance(n_steps,int)
        assert n_steps > 0
        self.n_steps = n_steps
        self.current_step = 0

        assert isinstance(time_step,float) or isinstance(time_step,int)
        self.time_step = float(time_step)

        assert isinstance(max_speed,float) or isinstance(max_speed,int)
        self.max_speed = float(max_speed)

        assert isinstance(snapshot_step,float) or isinstance(snapshot_step,int)
        self.snapshot_step = snapshot_step

        assert (seed is None) or isinstance(seed,int)
        self.seed = seed
        random.seed(self.seed)

        assert isinstance(n_jobs,int) and (n_jobs <= mp.cpu_count())
        self.n_jobs = n_jobs
        if (self.n_jobs > self.n_cells):
            self.n_jobs = self.n_cells

        self.parallel_char_min = parallel_char_min

        assert isinstance(output_path,str)
        self.output_path = output_path + datetime.today().strftime('%Y.%m.%d.%H.%M.%S') + '/'

        self.cell_dict = self.__generate_blank_cell_dict__()

        assert isinstance(max_characters,int)
        self.max_characters = max_characters
        self.n_characters = 0

        self.kill_early = kill_early

        self.__generate_linked_list__()

    def __str__(self):
        this_dict = self.__dict__
        out_str = "SimulationBox Object:\n"
        for key in this_dict:
            if (key!='cell_dict'):
                out_str += "\t"+key+":\t"+str(this_dict[key])+"\n"
        return out_str

    # Provides dict of 1D cells, value is list of all neighboring cells
    # Additionallly, generates offset list, so cell can appropriately add open boundaries
    def __generate_linked_list__(self):
        self.linked_list = {}
        self.boundary_offset_list = {}
        for x_n in range(0,self.n_cells):
            for y_n in range(0,self.n_cells):
                cell_index = self.convert_cell_2D_to_1D(x_n,y_n)
                self.linked_list[ cell_index ] = []
                self.boundary_offset_list[ cell_index ] = {}
                list_index = 0

                for x_n_neighbor in [x_n-1,x_n,x_n+1]:
                    for y_n_neighbor in [y_n-1,y_n,y_n+1]:
                        self.linked_list[cell_index].append(
                            self.convert_cell_2D_to_1D(
                                x_n_neighbor % self.n_cells,
                                y_n_neighbor % self.n_cells
                            )
                        )

                        x_off = 0.0
                        if ( x_n_neighbor < 0.0 ):
                            x_off =-1.0
                        elif ( x_n_neighbor >= self.n_cells ):
                            x_off = 1.0
                        y_off = 0.0
                        if ( y_n_neighbor < 0.0 ):
                            y_off =-1.0
                        elif ( y_n_neighbor >= self.n_cells ):
                            y_off = 1.0

                        self.boundary_offset_list[cell_index][list_index] = x_off*self.length,y_off*self.length
                        list_index += 1

    def __generate_blank_cell_dict__(self):
        cell_dict = {}
        for x_n in range(0,self.n_cells):
            for y_n in range(0,self.n_cells):
                cell_dict[self.convert_cell_2D_to_1D(x_n,y_n)] = []
        return cell_dict

    def list_params(self):
        return self.__dict__.keys()

    def get_param(self,name):
        assert name in self.__dict__
        return self.__dict__[name]

    def convert_cell_1D_to_2D(self,cell):
        y_cell = cell // self.n_cells
        x_cell = cell % self.n_cells
        return x_cell, y_cell

    def convert_cell_2D_to_1D(self,x_cell,y_cell):
        return self.n_cells * y_cell + x_cell

    # [[1 3][0 2]]
    def get_cell_1D(self,x,y):
        x_cell, y_cell = self.get_cell_2D(x,y)
        return self.convert_cell_2D_to_1D( x_cell, y_cell )

    # [[0,1 1,1][0,0 1,0]]
    def get_cell_2D(self,x,y):
        x_cell = int(x / self.length * self.n_cells)
        y_cell = int(y / self.length * self.n_cells)
        return x_cell, y_cell

    def get_boundary_offset(self,cell_number,i_tracker):
        return self.boundary_offset_list[cell_number][i_tracker]

    def box_is_full(self):
        return self.n_characters >= self.max_characters

    def embed(self,inp_char):
        assert isinstance(inp_char,characters.Character) or issubclass(inp_char,characters.Character)
        if (self.box_is_full()):
            return
        x, y = inp_char.get_pos()
        cell_index = self.get_cell_1D( x, y )
        self.cell_dict[cell_index].append(inp_char)
        self.n_characters += 1

    def update_solo_position(self,character):
        x, y = character.get_pos()
        s = character.get_speed() * self.time_step
        theta = character.get_orientation()
        new_x = ( x + s * np.cos(theta) ) % self.length
        new_y = ( y + s * np.sin(theta) ) % self.length
        return new_x, new_y

    # TODO: implement "mouth" based on orientation
    def check_collisions_feed(self,inp_char,new_x,new_y,cell_number,new_position_dict):
        # Ignore checking against self
        checked_ids = [inp_char.get_param('id')]
        consumed_chars = []

        ll_cell_tracker = 0
        for neighbor_cell_linked in self.linked_list[cell_number]:

            # Offsets due to open box
            x_offset, y_offset = self.get_boundary_offset(cell_number,ll_cell_tracker)
            ll_cell_tracker += 1

            # Check updated positions first, then non-updated
            for pos_dict in [new_position_dict,self.cell_dict]:

                for character in pos_dict[neighbor_cell_linked]:
                    # Ignore repeats, IE already updated
                    if ( character.get_param('id') in checked_ids ):
                        continue

                    checked_ids.append(character.get_param('id'))
                    overlap_val = overlap(
                        character.get_param('x') + x_offset,
                        character.get_param('y') + y_offset,
                        character.get_param('radius'),
                        inp_char.get_param('x'),
                        inp_char.get_param('y'),
                        inp_char.get_param('radius')
                    )
                    # No collision
                    if (overlap_val == 0):
                        continue

                    # Would overlap, both particles have collision
                    if (
                        (overlap_val == 2) and
                        inp_char.get_param('collision') and
                        character.get_param('collision')
                    ):
                        #TODO: this is broken
                        inp_char.speed.value = distance_two_circles(
                            character.get_param('x') + x_offset,
                            character.get_param('y') + y_offset,
                            character.get_param('radius'),
                            inp_char.get_param('x'),
                            inp_char.get_param('y'),
                            inp_char.get_param('radius')
                        ) / self.time_step
                        new_x, new_y = self.update_solo_position(inp_char)

                    # Eat if it's a food source for the moved particle
                    if (
                        ('eats' in inp_char.list_params()) and
                        (inp_char.get_param('eats')) and
                        (character.get_name()==inp_char.get_param('food_source'))
                    ):
                        inp_char.eat()
                        if (character.get_param('consumed')):
                            consumed_chars.append(character.get_param('id'))

        return new_x, new_y, consumed_chars

    def update_position_by_cell(self):
        position_changed = False
        new_position_dict = self.__generate_blank_cell_dict__()
        all_consumed_chars = []
        for cell_number in self.cell_dict:
            cell = self.cell_dict[cell_number]
            for c in cell:
                if ( 'speed' not in c.list_params() ):
                    new_position_dict[cell_number].append(c)
                else:
                    new_x, new_y = self.update_solo_position(c)
                    coll_x, coll_y, consumed_chars = self.check_collisions_feed(c,new_x,new_y,cell_number,new_position_dict)
                    coll_x = coll_x % self.length
                    coll_y = coll_y % self.length
                    c.update_pos( coll_x, coll_y )
                    new_cell = self.get_cell_1D( coll_x, coll_y )
                    new_position_dict[new_cell].append(c)
                    all_consumed_chars += consumed_chars
                    if (
                        (coll_x != c.get_param('x')) or
                        (coll_y != c.get_param('y'))
                    ):
                        position_changed |= True

        for cell_number in new_position_dict:
            cell = new_position_dict[cell_number]
            cell_w_removed = []
            for character in cell:
                if ( character.get_param('id') not in all_consumed_chars ):
                    cell_w_removed.append(character)
            self.cell_dict[cell_number] = cell_w_removed
        self.n_characters -= len(all_consumed_chars)

        return position_changed

    def update_cell_age(self,cell):
        survived_list = []
        any_age_changed = False
        n_killed = 0
        for char in cell:
            if ( 'age' in char.list_params() ):
                any_age_changed |= True
                if ( char.age_character(self.time_step) ):
                    survived_list.append(char)
                else:
                    n_killed += 1
            else:
                survived_list.append(char)
        
        return any_age_changed, survived_list, n_killed

    def update_cell_energy(self,cell):
        survived_list = []
        any_energy_changed = False
        n_killed = 0
        for char in cell:
            if ( 'energy' in char.list_params() ):
                any_energy_changed |= True
                if ( char.use_energy(self.time_step) ):
                    survived_list.append(char)
                else:
                    n_killed += 1
            else:
                survived_list.append(char)
        return any_energy_changed, survived_list, n_killed

    def update_cell_action(self,cell):
        action_changed = False
        acted_cell = []
        for char in cell:
            acted_cell.append( char )
            if (
                ('interprets' in acted_cell[-1].list_params()) and
                acted_cell[-1].get_param('interprets')
            ):
                acted_cell[-1].act(self.time_step)
                action_changed |= True
        return action_changed, acted_cell

    def update_cell_vision(self,cell,cell_number):
        vision_changed = False
        vision_cell = []
        for char in cell:
            vision_cell.append( char )
            if (
                ('vision' not in char.list_params()) or
                (not char.get_param('vision'))
            ):
                continue

            vision_cell[-1].get_param('eyes').reset_vision(0.0)
            ll_cell_tracker = 0
            for neighbor_cell_linked in self.linked_list[cell_number]:
                # Offsets due to open box
                x_offset, y_offset = self.get_boundary_offset(cell_number,ll_cell_tracker)
                ll_cell_tracker += 1
                for visible_obj in self.cell_dict[neighbor_cell_linked]:

                    if (vision_cell[-1].get_param('id') == visible_obj.get_param('id')):
                        continue

                    obj_dist = distance_two_circles(
                        vision_cell[-1].get_param('x'),
                        vision_cell[-1].get_param('y'),
                        vision_cell[-1].get_param('radius'),
                        visible_obj.get_param('x') + x_offset,
                        visible_obj.get_param('y') + y_offset,
                        visible_obj.get_param('radius')
                    )

                    # Only check vision lines if close enough
                    if ( vision_cell[-1].get_param('eyes').get_param('max_dist') >= obj_dist ):
                        ang_obj_char = ( relative_angle_between_characters( vision_cell[-1], visible_obj ) - vision_cell[-1].get_param('orientation').value ) % (2*math.pi)
                        left_obj_angle  = ang_obj_char + math.atan2( visible_obj.get_param('radius'), obj_dist )
                        right_obj_angle = ang_obj_char + math.atan2(-visible_obj.get_param('radius'), obj_dist )
                        vision_cell[-1].get_param('eyes').place_in_vision(visible_obj.get_name(),obj_dist,left_obj_angle,right_obj_angle)
                        vision_changed |= True
        return vision_changed, vision_cell

    def update_cell_spawn(self,cell,cell_number):
        any_spawned = False
        n_spawn_attempts = 5
        spawn_list = []
        for char in cell:
            if self.box_is_full():
                continue
            if (
                ('reproduces' in char.list_params()) and
                char.get_param('reproduces') and
                char.can_spawn(self.time_step)
            ):
                spawned = False
                # Try to spawn behind, to avoid collisions
                init_orientation = ( char.get_orientation() + math.pi ) % (2*math.pi)
                # Attempt to spawn 10 times, checking for collision
                for i in range(0,n_spawn_attempts):
                    if spawned:
                        break
                    for sign in [-1.,1.]:
                        if spawned:
                            break
                        spawn_angle = ( init_orientation + sign * i / n_spawn_attempts * math.pi / 2. ) % (2*math.pi)
                        spawn_dist  = 2.1 * char.get_param('radius')
                        x = ( char.get_param('x') + spawn_dist * math.cos(spawn_angle) ) % self.length
                        y = ( char.get_param('y') + spawn_dist * math.sin(spawn_angle) ) % self.length

                        child_cell = self.get_cell_1D(x,y)

                        no_collisions = True

                        ll_cell_tracker = 0
                        for neighbor_cell_linked in self.linked_list[cell_number]:

                            # Offsets due to open box
                            x_offset, y_offset = self.get_boundary_offset(child_cell,ll_cell_tracker)
                            ll_cell_tracker += 1

                            for neighbor in self.cell_dict[neighbor_cell_linked]:
                                overlap_val = overlap(
                                    neighbor.get_param('x') + x_offset,
                                    neighbor.get_param('y') + y_offset,
                                    neighbor.get_param('radius'),
                                    x,
                                    y,
                                    char.get_param('radius')
                                )

                                if (overlap_val!=0):
                                    no_collisions = False
                                if (not no_collisions):
                                    break

                            if (not no_collisions):
                                break

                        if no_collisions:
                            spawn_list.append( char.spawn( x, y ) )
                            spawned=True
                            any_spawned |= True
        return any_spawned, spawn_list

    def run_cell_operations(self,cell_number):
        cell = self.cell_dict[cell_number]

        changed_a, a_cell, n_killed_a = self.update_cell_age( cell )
        changed_v, v_cell             = self.update_cell_vision( a_cell, cell_number )
        changed_x, x_cell             = self.update_cell_action( v_cell ) # Update direction, speed
        changed_s, s_cell             = self.update_cell_spawn ( x_cell, cell_number )
        changed_e, e_cell, n_killed_e = self.update_cell_energy( x_cell )

        something_changed = any([
            changed_a,
            changed_v,
            changed_x,
            changed_s,
            changed_e,
        ])
        
        return cell_number, something_changed, e_cell, n_killed_a + n_killed_e, s_cell

    def iterate_characters(self):
        run_parallel = (self.n_jobs>1) and (
            (self.parallel_char_min is None) or
            (self.parallel_char_min >= self.n_characters)
        )

        # Run a set of operations in parallel if possible
        iter_list = []
        if (run_parallel):
            pool = ThreadPool(self.n_jobs)
            pool.restart()
            pool_results = pool.amap(
                self.run_cell_operations,
                [cell_number for cell_number in self.cell_dict.keys()]
            )
            pool.close()
            pool.join()
            iter_list = pool_results.get()
        else:
            for cell_number in self.cell_dict.keys():
                iter_list.append( self.run_cell_operations(cell_number) )

        # Bring together results and update cells
        something_changed = False
        spawn_list = []
        for cell_num, changed, updated_cell, n_killed, spawn_cell in iter_list:
            something_changed |= changed
            self.cell_dict[cell_num] = updated_cell
            self.n_characters -= n_killed
            spawn_list += spawn_cell
        for child in spawn_list:
            self.embed( child )

        # Final step move, this cannot be easily parallelized
        something_changed |= self.update_position_by_cell() #Feeds if collides with food source

        return something_changed

    def generate_snapshots(self):
        output_path = self.output_path + 'character_snapshots/'
        os.makedirs(output_path,exist_ok=True)

        output_fn = 'character_snapshot_{:09d}.pkl'.format(self.current_step)
        with open(output_path+output_fn,'wb') as f:
            for key in sorted(self.cell_dict.keys()):
                for c in self.cell_dict[key]:
                    pkl.dump(c.get_pickle_obj(),f)

        output_fn = 'simple_snapshot_{:09d}.pkl'.format(self.current_step)
        with open(output_path+output_fn,'wb') as f:
            for key in sorted(self.cell_dict.keys()):
                for c in self.cell_dict[key]:
                    out_dict = {
                        'id':c.get_param('id'),
                        'name':c.get_name(),
                        'x':c.get_param('x'),
                        'y':c.get_param('y'),
                        'size':c.get_param('size'),
                        'speed':c.get_speed(),
                        'orientation':c.get_orientation(),
                        'energy':c.get_energy(),
                        'age':c.get_age(),
                    }
                    pkl.dump(out_dict,f)
        #DEBUG
        #with open(output_path+"debug.txt",'a') as f:
        #    for key in sorted(self.cell_dict.keys()):
        #        for c in self.cell_dict[key]:
        #            out_str = (
        #                'id='+str(c.get_param('id'))+","+
        #                'name='+str(c.get_name())+","+
        #                'x='+str(c.get_param('x'))+","+
        #                'y='+str(c.get_param('y'))+","+
        #                'speed='+str(c.get_speed())+","+
        #                'size='+str(c.get_param('size'))+","+
        #                'orientation='+str(c.get_orientation())+","+
        #                'energy='+str(c.get_energy())+","+
        #                'age='+str(c.get_age())+"\n"
        #            )
        #            f.write(out_str)

    def iterate_step(self):
        something_changed = self.iterate_characters()
        if (self.current_step % self.snapshot_step == 0 ):
            self.generate_snapshots()
        return something_changed

    def run_simulation(self):
        start_time = datetime.now()
        prev_time = start_time
        for i in range(0,self.n_steps):
            something_changed = self.iterate_step()

            if (self.current_step % self.snapshot_step == 0 ):
                now = datetime.now()
                s = (now-start_time).total_seconds()
                hours, remainder = divmod(s, 3600)
                minutes, seconds = divmod(remainder, 60)
                print(
                    "Finished step {:09d}, time from last = {:4.1f} s, total time = {:02d}:{:02d}:{:03.1f}".format(
                        self.current_step,
                        (now-prev_time).total_seconds(),
                        int(hours),
                        int(minutes),
                        seconds
                    )
                )
                prev_time=now

            if (self.kill_early and (not something_changed)):
                break

            self.current_step = i + 1
