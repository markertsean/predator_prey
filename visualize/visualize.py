#import plotly.express as px
#import plotly.graph_objects as go
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl
import os
import sys

sys.path.append('/'.join( os.getcwd().split('/') )+'/')

import characters.characters as characters
import characters.parameters as parameters

def read_setup(input_path):
    output_dict = {}
    with open(input_path+"setup.log",'r') as f:
        for line in f:
            l = line.replace('\n','').split(':')
            try:
                if (l[1]=='False'):
                    l[1]=0
                elif (l[1]=='True'):
                    l[1]=1
                else:
                    numeric = any([x.isdigit() for x in l[1]])
                    if numeric:
                        l[1]=float(l[1])
                output_dict[l[0]] = l[1]
            except:
                pass
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

def read_character(input_path,input_params):
    all_files = os.listdir(input_path)
    char_files = []
    for file in all_files:
        if (file.startswith('character_snapshot_')):
            char_files.append(file)
    char_files = sorted(char_files)
    df_dict = {}
    static_dict = {}
    for fn in char_files:
        snap_str = ''.join(x for x in fn if x.isdigit())
        snap_int = int(snap_str)
        timestep = snap_int * input_params['time_step']

        for char in load_multi_pickle(input_path+fn):
            char_dict = {}
            for key,value in char:
                char_dict[key] = value
                
            char_name = char_dict['name']
            if (char_name not in df_dict):
                df_dict[char_name] = {}
                df_dict[char_name]['time'] = []

            df_dict[char_name]['time'].append(timestep)

            static_dict[char_dict['id']] = {}
            for key in char_dict.keys():
                # Check from good parameters
                if (
                    (key in [
                        'consumed',
                        'collision',
                        'size',
                        'radius',
                        'food',
                        'generation',
                    ]) or
                    (key.startswith('eye') and ('value' not in key)) or
                    (key.startswith('brain'))
                ):
                    static_dict[char_dict['id']][key] = char_dict[key]
                    continue

                if (key not in df_dict[char_name]):
                    df_dict[char_name][key] = []

                df_dict[char_name][key].append(char_dict[key])

    out_df_dict = {}
    for obj in df_dict.keys():
        this_dict = df_dict[obj]
        out_df_dict[obj] = pd.DataFrame(this_dict)
        
    return out_df_dict, static_dict

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

def plot_express(inp_df,setup_params):
    xy_max = setup_params['box_size']
    cell_size = setup_params['cell_size']
    fig = px.scatter(
        inp_df,
        x='x',
        y='y',
        animation_frame='time',
        animation_group='id',
        color='name',
        hover_name='id',
        hover_data=['speed','orientation','age','energy'],
        log_x=False,
        size='size',
        size_max=55,
        range_x=[0,xy_max],
        range_y=[0,xy_max],
    )
    fig.update_xaxes(tick0=cell_size, dtick=cell_size)
    fig.update_yaxes(tick0=cell_size, dtick=cell_size)
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1000*setup_params['time_step'] * setup_params['snapshot_step']
    fig.show()

def plot_go(inp_df,setup_params):
    xy_max = setup_params['box_size']
    cell_size = setup_params['cell_size']

    fig = go.Figure()

    char_list = []
    char_list.append(
        {
            'name':'food source',
            #'size':setup_params[],
        }
    )

    for char_dict in char_list:
        this_df = inp_df.loc[inp_df['name']==char_dict['name']].copy()

        fig.add_trace(
            go.Scatter(
                x=this_df['x'],
                y=this_df['y'],
                mode='markers',
                name='markers',
            )
        )
    
    fig.update_xaxes(tick0=cell_size, dtick=cell_size)
    fig.update_yaxes(tick0=cell_size, dtick=cell_size)
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1000*setup_params['time_step']
    fig.show()

def main():
    project_path = '/'.join(os.getcwd().split('/'))+'/'
    data_path = project_path + 'data/'

    input_version = 'latest'
    if (input_version=='latest'):
        input_version = sorted(os.listdir(data_path))[-1]
    input_base_path = data_path + input_version + '/'
    input_snap_path = input_base_path + 'character_snapshots/'

    setup_params = read_setup(input_base_path)
    #position_df  = read_simple(input_snap_path,setup_params)
    character_df_dict, static_dict  = read_character(input_snap_path,setup_params)

    # Food sources extend over boundaries, need to include in plot
    #position_wrapped_food_df = extend_food_sources(position_df)

    #plot_express(position_wrapped_food_df,setup_params)
    #plot_go(position_wrapped_food_df,setup_params)

if __name__ == '__main__':
    main()
