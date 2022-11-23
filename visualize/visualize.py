import plotly.express as px
import numpy as np
import pandas as pd
import pickle as pkl
import os

def read_setup(input_path):
    output_dict = {}
    with open(input_path+"setup.log",'r') as f:
        for line in f:
            l = line.replace('\n','').split(':')
            output_dict[l[0]] = float(l[1])
    return output_dict

def load_multi_pickle(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pkl.load(f)
            except EOFError:
                break

def read_simple(input_path,input_params):
    all_files = os.listdir(input_path)
    char_files = []
    for file in all_files:
        if (file.startswith('simple_snapshot_')):
            char_files.append(file)
    char_files = sorted(char_files)
    out_dict = {}
    for fn in char_files:
        snap_str = ''.join(x for x in fn if x.isdigit())
        snap_int = int(snap_str)
        timestep = snap_int * input_params['time_step']
        for this_dict in load_multi_pickle(input_path+fn):
            if len(out_dict.keys()) < 1:
                for key in this_dict:
                    out_dict[key] = []
                out_dict['time'] = []

            for key in this_dict:
                out_dict[key].append(this_dict[key])
            out_dict['time'].append(timestep)
    return pd.DataFrame.from_dict(out_dict)


def main():
    project_path = '/'.join(os.getcwd().split('/'))+'/'
    data_path = project_path + 'data/'

    input_version = 'latest'
    if (input_version=='latest'):
        input_version = sorted(os.listdir(data_path))[-1]
    input_base_path = data_path + input_version + '/'
    input_snap_path = input_base_path + 'character_snapshots/'

    setup_params = read_setup(input_base_path)
    position_df  = read_simple(input_snap_path,setup_params)

    fig = px.scatter(
        position_df,
        x='x',
        y='y',
        animation_frame='time',
        animation_group='id',
        color='name',
        hover_name='id',
        hover_data=['speed','orientation'],
        log_x=False,
        size_max=55,
        range_x=[0,1],
        range_y=[0,1],
    )
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1000*setup_params['time_step']
    fig.show()

if __name__ == '__main__':
    main()
