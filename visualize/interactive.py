from bokeh.io import curdoc
from bokeh.plotting import figure, show
from bokeh.layouts import column, row, widgetbox
from bokeh.models import Range1d, ColumnDataSource, Button, Slider, TapTool, glyphs, AnnularWedge, Wedge
from bokeh.events import Tap
from bokeh.colors import HSL,RGB, Color
import warnings
import colorsys
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

warnings.filterwarnings("ignore",category=DeprecationWarning)

'''
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

'''
def dict_time(inp,time_step):
    return round(inp,int(-np.log10(time_step)))

class CharMarker(glyphs.Circle):
    def __init__(self,color,radius,orientation,speed,eye_angle):
        pass

class Visualizer:
    def __init__(self,simulation_params,food_source_obj_df,prey_pred_obj_df,static_dict,brain_dict,max_time=None):
        self.play_button = Button(label="Play")
        self.callback = None

        self.static_dict = static_dict
        self.brain_dict = brain_dict

        self.current_time = 0.0
        self.time_step = simulation_params['time_step'] * simulation_params['snapshot_step']
        if (max_time==None):
            self.max_time = simulation_params['max_steps'] * simulation_params['time_step']
        else:
            assert isinstance(max_time,(int,float))
            self.max_time=max_time
        self.time_slider = Slider(
            start=0.0,
            end=self.max_time,
            value=0.0,
            step=self.time_step,
            title="Time:"
        )

        self.food_source_bokeh_input = ColumnDataSource(food_source_obj_df)

        colors =  {'prey': (100,255,100),'predator': (255,100,100)}
        self.prey_df = prey_pred_obj_df.copy()
        self.prey_df["color"] = self.prey_df["name"].apply(lambda c: colors[c])

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

        slit_size = 10
        self.prey_df['ori_left' ] = ( self.prey_df['orientation'] * 180/np.pi + slit_size / 2. ) % 360
        self.prey_df['ori_right'] = ( self.prey_df['orientation'] * 180/np.pi - slit_size / 2. ) % 360
        self.prey_iter_df = self.prey_df.loc[self.prey_df['time']==self.current_time].copy()

        self.prey_data_cols = ['id','x','y','speed','ori_left','ori_right','radius','color']
        self.prey_fig = self.fig.annular_wedge(
            source=ColumnDataSource(
                self.prey_iter_df[self.prey_data_cols]
            ),
            x='x',
            y='y',
            inner_radius_units='data',
            inner_radius=0.0,
            outer_radius_units='data',
            outer_radius='radius',
            start_angle_units='deg',
            end_angle_units='deg',
            start_angle='ori_left',
            end_angle='ori_right',
            fill_color = 'color',
            fill_alpha = 1.0,
            line_alpha = 0.0
        )

        self.examine_id = self.prey_iter_df.loc[0]['id']
        self.brain_scale_dict = None
        self.brain_plot = figure(
            title="Brain for id="+str(self.examine_id),
            x_axis_label='',
            y_axis_label='',
            sizing_mode='scale_height',
        )
        self.brain_bar_plot = None
        self.brain_circle_plot = None
        self.gen_brain_plot()

    def gen_char_markers(self,inp_df,color):
        pass

    def gen_char_marker(self,color,radius,orientation,speed,eye_angle):
        pass

    def get_brain_vals(self,inp_id,inp_name):
        name = inp_name + "_df"
        time_name = inp_name + "_iter_df"
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

    def gen_brain_plot(self):
        name = 'prey'
        if ( self.examine_id not in self.__dict__[name+"_iter_df"]['id'].unique() ):
            return
        self.brain_order = self.static_dict[self.examine_id]['brain_field_order']
        self.brain_data, self.brain_input_vals = self.get_brain_vals(self.examine_id,name)

        this_static_dict = self.static_dict[self.examine_id]

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

        max_height = 0
        input_layers = []

        for layer_size in [this_static_dict['brain_inputs']]+this_static_dict['brain_layer_sizes']:
            if ( layer_size > max_height ):
                max_height = layer_size
            input_layers.append(layer_size)

        n_plot_layers = len(input_layers)

        d_time = dict_time(self.current_time,self.time_step)
        #TODO: this is a bandaid solution to a bug where one obj class has a brain and other dont
        try:
            this_nn = self.brain_dict[d_time][self.examine_id]
        except:
            return

        # Set up brain colors based on height in graph, so is rainbow
        for i in range(0,this_static_dict['brain_inputs']):
            #r,g,b = colorsys.hls_to_rgb( float(i)/this_static_dict['brain_inputs'], 0.5, 0.9 )
            #bar_colors.append( RGB(int(r*255),int(g*255),int(b*255)) )
            bar_colors.append(HSL(i*360./this_static_dict['brain_inputs'],0.9,0.5))
            node_colors[-1].append(bar_colors[-1].darken(0.5).lighten(self.brain_data[i]*0.66))

        # Place coordinates centered around y=0, set colors based on node values and weights
        this_layer = None
        for i_x in range(0,n_plot_layers):
            if (i_x>0):
                this_layer = this_nn.get_layer(i_x-1)
                node_colors.append([])

                node_values.append(this_layer.calc(node_values[i_x-1]))

            for i_y in range(0,input_layers[i_x]):
                x_net.append( x_node_step * i_x + ( x_scale * ( x_min + 1 ) + x_node_off ) )
                y_net.append( -(i_y - input_layers[i_x]/2. + 0.5) )

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

                    weight_sum += 1e-10
                    r = int( r_tot / weight_sum )
                    g = int( g_tot / weight_sum )
                    b = int( b_tot / weight_sum )
                    node_colors[-1].append(RGB(r,g,b))

        node_colors_1d = []
        for i in range(0,len(node_colors)):
            for j in range(0,len(node_colors[i])):
                node_colors_1d.append( node_colors[i][j] )

        # Perform plot updates
        self.brain_plot.title.text="Brain for id="+str(self.examine_id)

        bar_df = pd.DataFrame({
            'val': x_scale * self.brain_data,
            'colors': bar_colors,
            'heights': [ bar_height for i in range(0,len(self.brain_data)) ],
            'y': [ max_height/2.-bar_height/2.-i for i in range(0,len(self.brain_data)) ],
        })

        if ( self.brain_bar_plot is None ):
            self.brain_bar_plot = self.brain_plot.hbar(
                source=bar_df,
                right='val',
                y = 'y',
                height = 'heights',
                color='colors'
            )
        else:
            self.brain_bar_plot.data_source.data = bar_df

        node_df = pd.DataFrame({
            'x': x_net,
            'y': y_net,
            'colors': node_colors_1d,
            'radius': [ node_size for i in range(0,len(x_net)) ],
        })

        if ( self.brain_circle_plot is None ):
            self.brain_circle_plot = self.brain_plot.circle(
                source=node_df,
                x = 'x',
                y = 'y',
                radius = 'radius',
                color = 'colors',
            )
        else:
            self.brain_circle_plot.data_source.data = node_df

    def update_chart(self):

        self.time_slider.value += self.time_step
        self.slider_callback('','','')

    def slider_callback(self,attr,old,new):

        self.time_slider.value %= self.max_time
        self.current_time = self.time_slider.value

        self.prey_iter_df = self.prey_df.loc[
            abs(self.prey_df['time']-self.time_slider.value) < 1e-4
        ]

        # Perform the data update
        self.prey_fig.data_source.data = (
            self.prey_iter_df[self.prey_data_cols]
        )
        self.gen_brain_plot()

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
        self.gen_brain_plot()

    def run_visualization(self):
        self.play_button.on_click(self.execute_animation)
        self.fig.on_event(Tap, self.click)
        self.time_slider.on_change('value',self.slider_callback)
        curdoc().add_root(row(column(self.fig,self.play_button,self.time_slider),self.brain_plot))



def generate_food_source_scatter_df(char_dict,static_dict):
    fs_df = char_dict['food_source'].drop(columns=['time']).drop_duplicates()
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

def generate_active_char_scatter_df(name,char_dict,static_dict,time_step):
    if name not in char_dict:
        return None, None
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
            new_df.loc[idx,param] = char[key]
    brain_cols = []
    for col in char_df.columns:
        if ('brain' in col):
            brain_cols.append(col)
    brain_dict = {}
    for t in sorted(char_df['time'].unique()):
        t_df = char_df.loc[char_df['time']==t]
        d_t = dict_time(t,time_step)
        brain_dict[d_t] = {}
        for idx,row in t_df.iterrows():
            i = row['id']
            if ( 'brain_weights' in row ):
                weights = row['brain_weights']
                biases  = row['brain_biases']
                afs     = static_dict[i]['brain_activation_functions']
                layers  = static_dict[i]['brain_layer_sizes']
                inputs  = static_dict[i]['brain_inputs']
                brain_dict[d_t][i] = NN.NeuralNetwork(
                    inputs,
                    layers,
                    weights,
                    biases,
                    afs,
                )
    return char_df.merge(new_df,on='id',how='inner').drop(columns=brain_cols), brain_dict

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

def animate_plot(simulation_params,char_dict,static_dict,max_time):
    food_source_df = generate_food_source_recursive_boundaries(char_dict, static_dict)
    time_dfs = []
    brain_dicts = {}
    for pp in ['prey','predator']:
        p_time_df, brain_dict = generate_active_char_scatter_df(pp,char_dict,static_dict,simulation_params['time_step'])
        if ( p_time_df is not None ):
            time_dfs.append( p_time_df )
            brain_dicts.update( brain_dict )

    time_df = pd.concat( time_dfs, ignore_index=True )

    my_visualizer = Visualizer(
        simulation_params,
        food_source_df,
        time_df,
        static_dict,
        brain_dicts,
        max_time = max_time
    )
    my_visualizer.run_visualization()


def main():

    project_path = '/'.join(os.getcwd().split('/'))+'/'
    data_path = project_path + 'data/character_snapshots/'

    input_version = 'latest'
    if (input_version=='latest'):
        input_version = sorted(os.listdir(data_path))[-1]
    input_log_path  = project_path + 'data/logfiles/' + input_version + '/'
    input_snap_path = data_path + input_version + '/'

    setup_params = viz.read_setup(input_log_path)
    character_df_dict, static_dict, max_timestep  = viz.read_character(input_snap_path,setup_params)

    animate_plot(setup_params,character_df_dict, static_dict, max_timestep)

main()

