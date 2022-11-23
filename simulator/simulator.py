from datetime import datetime
import numpy as np
import random
import pickle as pkl
import sys
import os

__project_path__='/'.join( __file__.split('/')[:-2] )+'/'
sys.path.append(__project_path__)

import characters.characters as characters

class SimulationBox:
    def __init__(
        self,
        box_size,
        cell_size,
        n_steps,
        time_step,
        max_speed,
        snapshot_step,
        seed,
        output_path = __project_path__+"data/",
    ):
        assert isinstance(box_size,float) or isinstance(box_size,int)
        self.length = float(box_size)

        assert isinstance(cell_size,float) or isinstance(cell_size,int)
        assert cell_size<self.length
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

        assert isinstance(output_path,str)
        self.output_path = output_path + datetime.today().strftime('%Y.%m.%d.%H.%M.%S') + '/'

        self.cell_dict = self.__generate_blank_cell_dict__()

        self.__generate_linked_list__()

    def __str__(self):
        this_dict = self.__dict__
        out_str = "SimulationBox Object:\n"
        for key in this_dict:
            if (key!='cell_dict'):
                out_str += "\t"+key+":\t"+str(this_dict[key])+"\n"
        return out_str

    # Provides dict of 1D cells, value is list of all neighboring cells
    def __generate_linked_list__(self):
        self.linked_list = {}
        for x_n in range(0,self.n_cells):
            for y_n in range(0,self.n_cells):
                cell_index = self.convert_cell_2D_to_1D(x_n,y_n)
                self.linked_list[ cell_index ] = []
                for x_n_neighbor in [x_n-1,x_n,x_n+1]:
                    for y_n_neighbor in [y_n-1,y_n,y_n+1]:
                        self.linked_list[cell_index].append(
                            self.convert_cell_2D_to_1D(
                                x_n_neighbor % self.n_cells,
                                y_n_neighbor % self.n_cells
                            )
                        )

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

    # [[0 1][2 3]] 
    def get_cell_1D(self,x,y):
        x_cell, y_cell = self.get_cell_2D(x,y)
        return self.convert_cell_2D_to_1D( x_cell, y_cell )

    # [[0,0 1,0][0,1 1,1]]
    def get_cell_2D(self,x,y):
        x_cell = int(x / self.length * self.n_cells)
        y_cell = int(y / self.length * self.n_cells)
        return x_cell, y_cell

    def embed(self,inp_char):
        assert isinstance(inp_char,characters.Character) or issubclass(inp_char,characters.Character)
        x, y = inp_char.get_pos()
        cell_index = self.get_cell_1D( x, y )
        self.cell_dict[cell_index].append(inp_char)

    def update_solo_position(self,character):
        x, y = character.get_pos()
        s = character.get_speed()
        theta = character.get_orientation()
        new_x = ( x + s * np.cos(theta) ) % self.length
        new_y = ( y + s * np.sin(theta) ) % self.length
        return new_x, new_y

    def update_position_by_cell(self):
        new_position_dict = self.__generate_blank_cell_dict__()
        for cell_number in self.cell_dict:
            cell = self.cell_dict[cell_number]
            for c in cell:
                if ( 'speed' not in c.list_params() ):
                    new_position_dict[cell_number].append(c)
                else:
                    new_x, new_y = self.update_solo_position(c)
                    c.update_pos( new_x, new_y )
                    new_cell = self.get_cell_1D( new_x, new_y )
                    new_position_dict[new_cell].append(c)

        self.cell_dict = new_position_dict

    def update_age_by_cell(self):
        survived_dict = self.__generate_blank_cell_dict__()
        for cell_number in self.cell_dict:
            cell = self.cell_dict[cell_number]
            for char in cell:
                if (
                    (not ('age' in char.list_params())) or
                    char.age_character(self.time_step)
                ):
                    survived_dict[cell_number].append(char)
        self.cell_dict = survived_dict

    def update_energy_by_cell(self):
        survived_dict = self.__generate_blank_cell_dict__()
        for cell_number in self.cell_dict:
            cell = self.cell_dict[cell_number]
            for char in cell:
                if (
                    (not ('energy' in char.list_params())) or
                    char.use_energy(self.time_step)
                ):
                    survived_dict[cell_number].append(char)
        self.cell_dict = survived_dict

    def iterate_characters(self):
        self.update_age_by_cell()
        self.update_energy_by_cell()
        # Eat
        self.update_position_by_cell()
        # Perception step
        # Update direction
        # Spawn step

    def generate_snapshots(self):
        output_path = self.output_path + 'character_snapshots/'
        os.makedirs(output_path,exist_ok=True)

        output_fn = 'character_snapshot_{:09d}.pkl'.format(self.current_step)
        with open(output_path+output_fn,'wb') as f:
            for key in sorted(self.cell_dict.keys()):
                for c in self.cell_dict[key]:
                    pkl.dump(c,f)

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


    def iterate_step(self):
        self.iterate_characters()
        if (self.current_step % self.snapshot_step == 0 ):
            self.generate_snapshots()

    def run_simulation(self):
        start_time = datetime.now()
        prev_time = start_time
        for i in range(0,self.n_steps):
            self.iterate_step()

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

            self.current_step = i + 1
