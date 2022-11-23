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

def extend_food_sources(inp_df):
    this_df = inp_df.copy()
    food_df = this_df.loc[this_df['name']=='food source']

    food_array = [this_df]
    for x in [-1,0,1]:
        for y in [-1,0,1]:
            if ( not( x==0 and y==0 ) ):
                temp_df = food_df.copy()
                temp_df['x'] = temp_df['x'] + x
                temp_df['y'] = temp_df['y'] + y
                temp_df['id']= temp_df['id'].astype(str)+"_"+str(x)+"_"+str(y)
                food_array.append(temp_df)
    return pd.concat(food_array)

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

    # Food sources extend over boundaries, need to include in plot
    position_wrapped_food_df = extend_food_sources(position_df)

    xy_max = setup_params['box_size']
    cell_size = setup_params['cell_size']
    fig = px.scatter(
        position_wrapped_food_df,
        x='x',
        y='y',
        animation_frame='time',
        animation_group='id',
        color='name',
        hover_name='id',
        hover_data=['speed','orientation','age','energy',],
        log_x=False,
        size='size',
        size_max=55,
        range_x=[0,xy_max],
        range_y=[0,xy_max],
    )
    fig.update_xaxes(tick0=cell_size, dtick=cell_size)
    fig.update_yaxes(tick0=cell_size, dtick=cell_size)
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1000*setup_params['time_step']
    fig.show()

if __name__ == '__main__':
    main()
