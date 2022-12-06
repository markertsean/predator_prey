from bokeh.io import curdoc
from bokeh.plotting import figure, show
from bokeh.layouts import column, row, widgetbox
from bokeh.models import Range1d, ColumnDataSource, Button, Slider
import numpy as np
import pandas as pd
import pickle as pkl
import os
import sys

print('/'.join(os.getcwd().split('/'))+'/')
sys.path.append('/'.join( os.getcwd().split('/') )+'/')

import characters.characters as characters
import characters.parameters as parameters
import visualize as viz

## prepare some data
#x = [1, 2, 3, 4, 5]
#y = [4, 5, 5, 7, 2]
#
## apply theme to current document
#curdoc().theme = "dark_minimal"
#
## create a plot
#p = figure(sizing_mode="stretch_width", tooltips="Data point @x has the value @y",max_width=500, height=250)
#
##p.y_range=Range1d(bounds=(0, 1))
#
## add a renderer
#p.line(x, y)
#
## show the results
#show(p)

'''
from bokeh.io import curdoc
from bokeh.plotting import figure, show
from bokeh.models import Range1d


source = ColumnDataSource(df)
p.circle(x='x_values', y='y_values', source=source)


# apply theme to current document
curdoc().theme = "dark_minimal"

p = figure(title="Multiple line example", x_axis_label="x", y_axis_label="y")
p.y_range=Range1d(bounds=(0, 1))

# add multiple renderers
p.line(x, y1, legend_label="Temp.", color="blue", line_width=2)

#p.circle(x, y3, legend_label="Objects", fill_color=(100,200,255), fill_alpha=0.5, line_color='red',radius=0.25, radius_units='data')
circle = p.circle(x, y3, legend_label="Objects", fill_color=(100,200,255), fill_alpha=0.5, line_color='red',radius=0.25, radius_units='data')

# change color of previously created object's glyph
glyph = circle.glyph
glyph.fill_color = (255,200,100)

colors = ["#%02x%02x%02x" % (255, int(round(value * 255 / 100)), 255) for value in y]
circle = p.circle(x, y, fill_color=colors, line_color="blue", size=15)


'''

class Visualizer:
    def __init__(self,simulation_params,food_source_obj_df,prey_obj_df):
        self.play_button = Button(label="Play")

        self.current_time = 0.0
        self.time_step = simulation_params['time_step'] * simulation_params['snapshot_step']
        self.max_time = 5.#simulation_params['max_steps'] * simulation_params['time_step']
        self.time_slider = Slider(
            start=0.0,
            end=self.max_time,
            value=0.0,
            step=self.time_step,
            title="Time:"
        )

        self.food_source_bokeh_input = ColumnDataSource(food_source_obj_df)

        self.prey_df = prey_obj_df.copy()
        self.prey_iter_df = self.prey_df.loc[self.prey_df['time']==self.current_time]

        self.fig = figure( title="Box", x_axis_label="x", y_axis_label="y", sizing_mode='scale_height')
        self.fig.x_range = Range1d( bounds=(0,1) )
        self.fig.y_range = Range1d( bounds=(0,1) )

        self.food_sources_fig = self.fig.circle(
            source=self.food_source_bokeh_input,
            x='x',
            y='y',
            radius_units='data',
            radius='radius',
            fill_color = (100,200,255),
            fill_alpha = 0.30,
            line_alpha = 0.0
        )

        self.prey_data_cols = ['id','x','y','speed','orientation','radius']
        self.prey_fig = self.fig.circle(
            source=ColumnDataSource(
                self.prey_iter_df[self.prey_data_cols]
            ),
            x='x',
            y='y',
            radius_units='data',
            radius='radius',
            fill_color = (100,255,100),
            fill_alpha = 1.0,
            line_alpha = 0.0
        )
        self.callback = None

    def update_chart(self):

        self.time_slider.value += self.time_step
        self.time_slider.value %= self.max_time
        self.current_time = self.time_slider.value

        self.prey_iter_df = self.prey_df.loc[
            abs(self.prey_df['time']-self.time_slider.value) < 1e-4
        ]
        print(self.time_slider.value,self.prey_iter_df.shape)
        self.prey_fig.data_source.data = (
            self.prey_iter_df[self.prey_data_cols]
        )

        if (self.current_time > self.max_time):
            self.current_time = 0.0

    def execute_animation(self):
        if (self.play_button.label == "Play"):
            self.play_button.label = "Pause"
            self.callback = curdoc().add_periodic_callback(self.update_chart,self.time_step * 1000)
        else:
            self.play_button.label = "Play"
            curdoc().remove_periodic_callback(self.callback)

    def run_visualization(self):
        self.play_button.on_click(self.execute_animation)
        curdoc().add_root(column(self.fig,self.play_button,self.time_slider))
        curdoc().add_root(self.time_slider)



def generate_food_source_scatter_df(char_dict,static_dict):
    fs_df = char_dict['food source'].drop(columns=['time']).drop_duplicates()
    id_list = fs_df['id'].unique()
    rad_list = []
    size_list = []
    consumed_list = []
    collision_list = []
    for idx in id_list:
        fs = static_dict[idx]
        rad_list.append      ( fs['radius'   ] )
        size_list.append     ( fs['size'     ] )
        consumed_list.append ( fs['consumed' ] )
        collision_list.append( fs['collision'] )
    fs_df['radius'   ] = rad_list
    fs_df['size'     ] = size_list
    fs_df['consumed' ] = consumed_list
    fs_df['collision'] = collision_list

    return fs_df

def generate_food_source_recursive_boundaries(char_dict,static_dict):
    fs_df = generate_food_source_scatter_df(char_dict, static_dict)

    df_list = []
    for x in [-1,0,1]:
        for y in [-1,0,1]:
            df_list.append( fs_df.copy() )
            df_list[-1]['x'] = df_list[-1]['x'] + x
            df_list[-1]['y'] = df_list[-1]['y'] + y
    return pd.concat(df_list)

def generate_active_char_scatter_df(name,char_dict,static_dict):
    char_df = char_dict[name]
    copy_fields = []
    new_df = pd.DataFrame(char_df['id'].copy())
    for param in [
        'radius',
    ]:
        copy_fields.append(param)
        new_df[param] = 0
    new_df = new_df.drop_duplicates()
    for idx, row in new_df.iterrows():
        this_id = row['id']
        char = static_dict[this_id]
        for key in copy_fields:
            ############TODO: Remove multiplier
            new_df.loc[idx,param] = char[key] * 10.0
    return char_df.merge(new_df,on='id',how='inner')

def output_static_plot(char_dict,static_dict):
    food_source_df = generate_food_source_recursive_boundaries(char_dict, static_dict)
    food_source_bokeh_input = ColumnDataSource(food_source_df)

    prey_time_df = generate_active_char_scatter_df('prey',char_dict,static_dict)

    t = 0.0
    prey_iter_df = prey_time_df.loc[prey_time_df['time']==t]
    prey_position_input = ColumnDataSource(prey_iter_df[['id','x','y','speed','orientation','radius']])

    plt = figure( title="Box", x_axis_label="x", y_axis_label="y", sizing_mode='scale_height')

    plt.x_range = Range1d( bounds=(0,1) )
    plt.y_range = Range1d( bounds=(0,1) )

    food_sources_plt = plt.circle(
        source=food_source_bokeh_input,
        x='x',
        y='y',
        radius_units='data',
        radius='radius',
        fill_color = (100,200,255),
        fill_alpha = 0.30,
        line_alpha = 0.0
    )

    prey_plt = plt.circle(
        source=prey_position_input,
        x='x',
        y='y',
        radius_units='data',
        radius='radius',
        fill_color = (100,255,100),
        fill_alpha = 1.0,
        line_alpha = 0.0
    )

    show(plt)

def animate_plot(simulation_params,char_dict,static_dict):
    food_source_df = generate_food_source_recursive_boundaries(char_dict, static_dict)
    prey_time_df = generate_active_char_scatter_df('prey',char_dict,static_dict)

    print(prey_time_df['time'].unique())
    my_visualizer = Visualizer(
        simulation_params,
        food_source_df,
        prey_time_df
    )
    my_visualizer.run_visualization()


def main():

    project_path = '/'.join(os.getcwd().split('/'))+'/'
    data_path = project_path + 'data/'

    input_version = 'latest'
    if (input_version=='latest'):
        input_version = sorted(os.listdir(data_path))[-1]
    input_base_path = data_path + input_version + '/'
    input_snap_path = input_base_path + 'character_snapshots/'

    setup_params = viz.read_setup(input_base_path)
    character_df_dict, static_dict  = viz.read_character(input_snap_path,setup_params)

    animate_plot(setup_params,character_df_dict, static_dict)

main()

