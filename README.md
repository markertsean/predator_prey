
# Predator Prey Simulator

This project is designed to evolve the behavior of predator and prey in a simple simulation, starting with random movements and decision making of actors, but allowing the actors to learn behaviors based on mutations and positive and negative reinforcement learning. The predator seek to hunt down prey, and the prey seek static food sources while avoiding predators. All the characters expend energy requiring they eat, have a finite lifetime, and can reproduce. This all takes place in a 2D simulation environment.

A more detailed write-up with examples can be found under my project page: https://markertsean.github.io/projects/



## Getting Started

This project was developed using Python 3.11, I wouldn't recommend using earlier versions of Python. I would also highly recommend using a virtual environment to install the dependencies, as it uses some now deprecated libraries for visualizations.

Download the project into a directory of your choice and install the requirements in dependencies.txt. On linux systems, you can run

```bash
  pip install -r requirements.txt
```

Before you run the simulation, you can edit the settings/config.py file. This is not a standard text based config file but is instead a python file, as I wanted an easy way to link the sizes of items within the simulation box to the simulation box size. You are able to edit the parameters as if it was a standard config file. To run a simulation on linux systems, in the project home directory you can run

```bash
  python run.py
```

This will run the simulation using the values in the config file, and create a data directory with snapshots of the simulation. To visualize the simulation on linux systems, in the project home directory you can run

```bash
  bokeh serve --show visualize/interactive.py
```

This will open an interactive bokeh server in a web browser. You can drag to a certain simulation time, and click on a character, in which case the window on the right will show the vision cones of a character and how it affects the decision making of the character (final outputs being acceleration and turning acceleration).

## Tunable parameters

### Simulation
* max_steps - Maximum until simulation is stopped
* time_step - Time step, lower values is higher resolution
* snapshot_step - How often to output a snapshot
* box_size - Box size, scales down other attributes
* cell_size - Tunable, only needs to be big enough for characters to see and react to other nearby characters
* abs_max_speed - Limit on all objects moving in box
* max_characters - Limit to avoid explosion in calculations
* kill_no_diff - Whether to end the simulation if the simulation state doesn't change
* seed - Random seed, None uses current time

### Age
* age - True/False, whether the character ages
* age_max - Maximum age of a character
### Energy
* energy - True/False, whether the character requires energy
* energy_max - Maximum energy a character can have. When eats, gains max energy.
* energy_time_delta - Amount of energy to subtract every second
* energy_speed_delta - Amount of energy to subtract every second, multiplied by the fraction of speed over max speed
### Food
* needs_food - True/False, whether the character eats
* food_source - Name of the food the character eats
* food_of - Name of what eats this character
### Vision
* vision - True/False, whether the character sees
* eye_offset - Angle in radians to center the vision rays
* eye_fov - Angle in radians providing the field of view on one side
* eye_rays - Number of rays to fill the vision FOV with
* eye_dist - How far out to measure vision, in simulation distance units
* eye_objs - List of names of characters the character can see
### Interpretation
* brain - True/False, whether to create a MLP to translate vision to speed/orientation changes
* brain_layers - List of number of nodes in each layer of the neural network
* brain_weights - List of list of weights for each layer of the neural network
* brain_biases - List of biases for each layer of the neural network
* brain_AF - List of activation functions for each layer
* complex_brain - True/False, whether to create more detailed neural network, * allowing arbitrary combinations of MLPs, other variables for decision making
* complex_brain_input_structure - List of strings containing character variable names, MLP files to include in the complex_brain
* complex_brain_variables - List of strings containing character variables names in order of input into the complex_brain network.
* complex_brain_nn_structure - List of number of nodes in each layer of the neural network that combines the input
* save_brains - True/False, whether to save the brain files
* brain_output_version - File name to save brain file
* brain_output_version_date - True/False, whether to append the date to the file name
* load_brains - True/False, whether to load the brain files
* brain_input_version - File name to load brain file
* learns - True/False, whether to use re-enforcement learning to adjust neural network weights
* learning_max - Maximum allowable learning rate in brain weights, at age 0
* learning_floor - Minimum allowable learning rate in brain weights, at age max
* learning_halflife - Number of generations before learning rate cut in half (+ minimum)
* X_move_reward - Reward multiplier for moving in the direction of object with name X while learning
* X_orientation_reward - Reward multiplier for turning in the direction of object with name X while learning
### Reproduction
* spawns_fixed - True/False, whether or not to create a duplicate of this character at fixed time intervals
* spawn_time_fixed - How long until spawn new character
* new_spawn_delay - Offset to allow extra time before first spawn
* spawn_energy_min - Minimum energy requirement for reproduction
* spawn_energy_delta - Energy to subtract when spawns
* mutation_max - Maximum allowable mutation rate in brain weights, at generation 0
* mutation_floor - Minimum allowable mutation rate in brain weights, at generation inf
* mutation_halflife - Number of generations before mutation rate cut in half (+ minimum)


## Simulation environment

The simulations box is an open box object that we embed the character object in. The open nature of the box allows characters to leave the boundaries on either side, IE if a character passes beyond the right boundary, it will appear on the left side of the box. We utilize a cell structure that restrict computations between characters to only those in the same or neighboring cells. This significantly reduces the number of calculations in the box, especially for vision and collision calculations. Additionally, the box imposes a maximum speed and maximum number of characters, to avoid explosions of calculations. The user can specify the timestep and runtime, as well as how often to output a snapshot. If there is no change between timesteps, the simulation will terminate.

The simulation box calls a set of character functions for each character, calling the function for all characters before moving to the next function. In order, these are: aging the characters, calculate the vision values, interpret the vision to update the speed and orientation, spawn new characters, update the character energy, and as a last step, move and eat. This can be parallelized to run concurrently by cell, with the exception of the "move" step. Every function prior to this one can be calculated independent of the other characters, but during the move step we check for collisions, and thus each character needs the up-to-date position to avoid colliding with a character that is no longer there.

## Character objects

The character objects are the representation of predator and prey in the simulation, where each is a subclass of the character class. The only difference is prey have an infinite static food source randomly distributed throughout the box, whereas the predators have a food source of the prey class.

The characters by default have a few intrinsic properties, including: positions x & y, radius (all are circle shaped), a name, and whether it is consumed (deleted) by what eats it. Additionally, there are a number of attributes that can be activated by the user, allowing extra functionality. These are listed in the tunable parameter list.

One of the vital character parameters is the vision. The vision object creates a set of rays oriented at different angles on either side of the characters orientation. The user can provide the center of these group of rays, and the total field of view. IE, if the user provides a FOV of 90 degrees, offset of 45 degrees, and 9 rays, the rays will all have a width of 10 degrees, and the "left" rays will be centered on (left to right) [85,75,65,55,45,35,25,15,5] degrees, and the "right" rays centered on [355,345,335,325,315,305,295,285,275] degrees. Any named object that the character can see that intersects one of these rays will be placed within it, and the closest values are used to determine changes in behavior.

## Behavioral training

With the vision object, we can pair it with a Multi Layered Perceptron to train the predator and prey behavior, by outputting changes to speed and orientation as the final layer. We do this using a simple MLP to ensure we capture intended behavior or turning towards food, avoiding collisions, and avoiding predators. We later those combine those with a MLP to construct a much stronger neural network that can weigh these decisions against each other.

This can be accomplished by turning on the brains in the config, enabling learning or mutations, and running the simulation for a long period of time. A current branch is under development that includes a pre-training enviornment, so brains can be trained for behavior and loaded in.
