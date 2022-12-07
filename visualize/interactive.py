from bokeh.io import curdoc
from bokeh.plotting import figure, show
from bokeh.layouts import column, row, widgetbox
from bokeh.models import Range1d, ColumnDataSource, Button, Slider, TapTool
from bokeh.events import Tap
from bokeh.colors import HSL, RGB
import numpy as np
import pandas as pd
import pickle as pkl
import os
import sys

sys.path.append('/'.join( os.getcwd().split('/') )+'/')

import characters.characters as characters
import characters.parameters as parameters
import perceptron.neural_net as NN
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
    def __init__(self,simulation_params,food_source_obj_df,prey_obj_df,static_dict):
        self.play_button = Button(label="Play")
        self.callback = None

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

        my_tools = 'pan,wheel_zoom,zoom_in,zoom_out,box_zoom,reset,tap'
        self.fig = figure( title="Box", x_axis_label="x", y_axis_label="y", sizing_mode='scale_height', tools=my_tools)
        self.fig.x_range = Range1d( bounds=(0,1) )
        self.fig.y_range = Range1d( bounds=(0,1) )
        self.taptool = self.fig.select(type=TapTool)

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


        
        self.examine_id = self.prey_iter_df.loc[0]['id']
        self.brain_scale_dict = None
        # Should always be the same
        self.brain_order = static_dict[self.examine_id]['brain_field_order']
        self.brain_data, self.brain_input_vals = self.get_brain_vals(self.examine_id,'prey_df','prey_iter_df')


        this_static_dict = static_dict[self.examine_id]

        max_height = 0
        input_layers = []
        for layer_size in [this_static_dict['brain_inputs']]+this_static_dict['brain_layer_sizes']:
            if ( layer_size > max_height ):
                max_height = layer_size
            input_layers.append(layer_size)
        
        n_plot_layers = len(input_layers)

        node_size   = 0.015
        bar_height  = 0.75
        x_node_off  = 0.1
        x_node_step = 0.5
        x_min       = 0
        x_scale     = 0.5
        x_net       = []
        y_net       = []
        bar_colors  = []
        node_colors = [[]]
        node_values = [self.brain_input_vals]

        this_nn = NN.NeuralNetwork(
            this_static_dict['brain_inputs'],
            layer_sizes = this_static_dict['brain_layer_sizes'],
            weights = this_static_dict['brain_weights'],
            biases = this_static_dict['brain_biases'],
            activation_functions = this_static_dict['brain_activation_functions'],
        )
        
        for i in range(0,this_static_dict['brain_inputs']):
            bar_colors.append(HSL(i*360./this_static_dict['brain_inputs'],0.9,0.5))
            node_colors[-1].append(bar_colors[-1].darken(1.0).lighten(self.brain_data[i]*0.66))

        this_layer = None
        for i_x in range(0,n_plot_layers):
            x_net.append([])
            y_net.append([])

            if (i_x>0):
                this_layer = this_nn.get_layer(i_x-1)
                node_colors.append([])

                node_values.append(this_layer.calc(node_values[i_x-1]))

            for i_y in range(0,input_layers[i_x]):
                x_net[-1].append( x_node_step * i_x + ( x_scale * ( x_min + 1 ) + x_node_off ) )
                y_net[-1].append( -(i_y - input_layers[i_x]/2. + 0.5) )

                r_tot=0
                g_tot=0
                b_tot=0
                
                if (i_x>0):
                    this_neuron = this_layer.get_neurons()[i_y]
                    weight_sum = 0
                    for i_w in range(0,len(this_neuron.get_weights())):
                        weight = this_neuron.get_weights()[i_w]
                        val    = node_values[i_x-1][i_w]
                        weight_sum += weight * val
                        
                        r_tot += weight * val * node_colors[i_x-1][i_w].to_rgb().r
                        g_tot += weight * val * node_colors[i_x-1][i_w].to_rgb().g
                        b_tot += weight * val * node_colors[i_x-1][i_w].to_rgb().b

                    r = int( r_tot / weight_sum )
                    g = int( g_tot / weight_sum )
                    b = int( b_tot / weight_sum )
                    node_colors[-1].append(RGB(r,g,b))

            
        self.brain_plot = figure(
            title="Brain for id="+str(self.examine_id),
            x_axis_label='',
            y_axis_label='',
            sizing_mode='scale_height',
        )
        self.brain_plot.hbar(
            right=x_scale * self.brain_data,
            y = [ max_height/2.-bar_height/2.-i for i in range(0,len(self.brain_data)) ],
            height = bar_height,
            color=bar_colors
        )
        for x_layer, y_layer, c_layer in zip( x_net, y_net, node_colors ):
            self.brain_plot.circle(
                x = x_layer,
                y = y_layer,
                radius = node_size,
                color = c_layer
            )

    def get_brain_vals(self,inp_id,name,time_name):
        this_all_df = self.__dict__[name]
        this_df = self.__dict__[time_name]
        this_id_data = this_df.loc[ this_df['id'] == inp_id ][self.brain_order].values[0]
        
        if ( self.brain_scale_dict is None ):
            self.brain_scale_dict = {}
        if ( name not in self.brain_scale_dict):
            self.brain_scale_dict[name] = {}

        out_array = np.zeros(len(self.brain_order))
        i = 0
        for col in self.brain_order:
            if ( col not in self.brain_scale_dict[name] ):
                self.brain_scale_dict[name][col] = this_all_df[col].max()

            out_array[i] = this_id_data[i] / (self.brain_scale_dict[name][col]+1e-7)
            i += 1

        return out_array, this_id_data

    def update_chart(self):

        self.time_slider.value += self.time_step
        self.time_slider.value %= self.max_time
        self.current_time = self.time_slider.value

        self.prey_iter_df = self.prey_df.loc[
            abs(self.prey_df['time']-self.time_slider.value) < 1e-4
        ]
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

    def click(self,event):
        x_click = event.x
        y_click = event.y

        min_dist = 1e10
        min_i = 0
        min_id = self.prey_fig.data_source.data['id'][min_i]
        
        for char_source in [self.prey_fig.data_source.data]:
            this_time_data = char_source
            for i in range(0,this_time_data['id'].shape[0]):
                this_dist = np.sqrt(
                    (x_click - this_time_data['x'][i])**2 + (y_click - this_time_data['y'][i])**2
                )
                if (min_dist>this_dist):
                    min_dist = this_dist
                    min_i = i
                    min_id = self.prey_fig.data_source.data['id'][min_i]

        self.examine_id = min_id
    
    def run_visualization(self):
        self.play_button.on_click(self.execute_animation)
        self.fig.on_event(Tap, self.click)
        curdoc().add_root(row(column(self.fig,self.play_button,self.time_slider),self.brain_plot))



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

    my_visualizer = Visualizer(
        simulation_params,
        food_source_df,
        prey_time_df,
        static_dict,
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

