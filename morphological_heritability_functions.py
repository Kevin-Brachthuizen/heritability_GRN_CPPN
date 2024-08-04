import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import multineat

from behavioural_heritability_functions import get_parent_ids, get_generation_data
from revolve2.ci_group.genotypes.cppnwin.modular_robot.v2._body_develop_grn import DevelopGRN
from revolve2.ci_group.genotypes.cppnwin.modular_robot.v2._body_develop import develop

"""
Some auxiliary functions that perform basic functions and operations
"""

def str_to_floatlist(string_genotype):
    """Simply converts a list of strings to a list of floats"""
    list_genotype = string_genotype.split(",")
    finallist = [float(i) for i in list_genotype]
    
    return finallist


def make_grid():
    """This function makes an empty grid to fill in"""
    grid = np.zeros((63,63))
    grid[31:34,31:34] = 1
    return grid


def change_dir(dir, rotation):
    """Rotates the direction to the new direction dictated by a new module"""
    #check if valid values
    if dir not in ['up', 'right', 'left', 'down']:
        raise ValueError("Not a valid value for direction (mus be either 'up', 'down', 'left' or 'right')")
    if rotation not in ['anti-clockwise', 'clockwise']:
        raise ValueError("Not a valid value for rotation (must be 'clockwise' or 'anti-clockwise')")

    #change direction according to rotation
    if (dir == 'up' and rotation == 'anti-clockwise') or (dir == 'down' and rotation == 'clockwise'):
        newdir = 'left'
    elif (dir == 'up' and rotation == 'clockwise') or (dir == 'down' and rotation == 'anti-clockwise'):
        newdir = 'right'
    elif (dir == 'right' and rotation == 'anti-clockwise') or (dir == 'left' and rotation == 'clockwise'):
        newdir = 'up'
    elif (dir == 'right' and rotation == 'clockwise') or (dir == 'left' and rotation == 'anti-clockwise'):
        newdir = 'down'
    
    return newdir


def change_coords(cur_coords, dir, steps=1):
    UP = np.array([0,1])
    RIGHT = np.array([1,0])
    DOWN = np.array([0,-1])
    LEFT = np.array([-1,0])

    #check if valid values
    if dir not in ['up', 'right', 'left', 'down']:
        raise ValueError("Not a valid value for direction (mus be either 'up', 'down', 'left' or 'right')")

    #move in direction depending on direction
    if dir == 'up':
        newcoords = np.add(cur_coords, UP*steps)
    elif dir == 'down':
        newcoords = np.add(cur_coords, DOWN*steps)
    elif dir == 'right':
        newcoords = np.add(cur_coords, RIGHT*steps)
    elif dir == 'left':
        newcoords = np.add(cur_coords, LEFT*steps)
    
    return newcoords


def give_coordinate_value(module, coordinates, grid):
    """This function will give a coordinate a value based on the module that is present there"""
    #warning if spot is not empty
    try:
        if grid[coordinates[0], coordinates[1]] != 0:
            print("Warning: Value already present -> Overwriting existing values")
    except IndexError:
        print("Warning: Coordinates out of bounds -> Skipping this module")
        return

    #rewrite the spot to a value that corresponds to each module type
    value = get_module_type(module)
    if value == 'brick':
        grid[coordinates[0], coordinates[1]] = 2
    elif value == 'hinge':
        grid[coordinates[0], coordinates[1]] = 3


# def get_bodies_from_id(offspring_id, df_population, df_genotype):
#     #get the right genotype and develop the body
#     offspring_genid = df_population['genotype_id'][offspring_id]
#     parent1_genid, parent2_genid = get_parent_ids(offspring_id, df_population)

#     offspring_gen = df_genotype['serialized_body'][offspring_genid-1]
#     genome = multineat.Genome()
#     genome.Deserialize(offspring_gen)
#     offspring = develop(genome, df_genotype['mapping_seed'][offspring_genid-1], False, False, False, False, 20, False, True, False, False, True)

#     #get the right genotype and develop the body of the parents if they exist
#     if parent1_genid != -1 and parent2_genid != -1:
#         parent1_gen = df_genotype['serialized_body'][parent1_genid]
#         parent2_gen = df_genotype['serialized_body'][parent2_genid]

#         genome1 = multineat.Genome()
#         genome1.Deserialize(parent1_gen)
#         genome2 = multineat.Genome()
#         genome2.Deserialize(parent2_gen)

#         parent1 = develop(genome1, df_genotype['mapping_seed'][parent1_genid-1], False, False, False, False, 20, False, True, False, False, True)
#         parent2 = develop(genome2, df_genotype['mapping_seed'][parent2_genid-1], False, False, False, False, 20, False, True, False, False, True)

#         return offspring, parent1, parent2
#     else:
#         return offspring, 'empty', 'empty'


def get_bodies_from_id(offspring_id, df_population, df_genotype):
    print(offspring_id)
    #get the right genotype and develop the body
    offspring_genid = df_population['genotype_id'][offspring_id]
    parent1_genid, parent2_genid = get_parent_ids(offspring_id, df_population)

    offspring_gen = df_genotype['serialized_body'][offspring_genid-1]

    offspring = str_to_floatlist(offspring_gen)
    offspringdev = DevelopGRN(20, True, offspring)
    offspringbody = offspringdev.develop()

    #get the right genotype and develop the body of the parents if they exist
    if parent1_genid != -1 and parent2_genid != -1:
        parent1_gen = df_genotype['serialized_body'][parent1_genid]
        parent2_gen = df_genotype['serialized_body'][parent2_genid]

        parent1 = str_to_floatlist(parent1_gen)
        parent2 = str_to_floatlist(parent2_gen)
        dev1 = DevelopGRN(20, True, parent1)
        dev2 = DevelopGRN(20, True, parent2)
        parent1body = dev1.develop()
        parent2body = dev2.develop()

        return offspringbody, parent1body, parent2body
    else:
        return offspringbody, 'empty', 'empty'
    

"""
Functions that are specific to getting the morphology and properties related to it
"""

def get_module_type(module):
    """Get the type of module that is present in the current spot"""
    if 'BrickV2' in str(module):
        return 'brick'
    elif 'HingeV2' in str(module):
        return 'hinge'


def write_branch(body, start_coord, start_dir, grid):
    """This function writes a branch to the grid, it assumes that you got past the core attachment point already"""
    #Set 'parent' value in the grid
    give_coordinate_value(body, start_coord, grid)

    #iterate over all children and write their values
    for key in body.children:
        child = body.children[key]
        #if we go straight we just move forward
        if key == 0:
            """front"""
            cur_dir = start_dir
            cur_coord = change_coords(start_coord, cur_dir)
        #If we go right or left we need to change direction as well
        elif key == 1:
            """right"""
            cur_dir = change_dir(start_dir, 'clockwise')
            cur_coord = change_coords(start_coord, cur_dir)
        elif key == 2:
            """left"""
            cur_dir = change_dir(start_dir, 'anti-clockwise')
            cur_coord = change_coords(start_coord, cur_dir)

        # Recursively go through entire branch
        write_branch(child, cur_coord, cur_dir, grid)


def get_filled_matrix(body):
    """This function will return a matrix with the filled in values of the body"""
    #make a grid to fill in
    grid = make_grid()
    start_coords = [32,32]

    #The core is already written, so we start from its children immediately
    for key in body.core_v2.children:
        if key == 0:
            direction = 'up'
            cur_coords = change_coords(start_coords, direction, 2)
        elif key == 1:
            direction = 'right'
            cur_coords = change_coords(start_coords, direction, 2)
        elif key == 2:
            direction = 'down'
            cur_coords = change_coords(start_coords, direction, 2)
        elif key == 3:
            direction = 'left'
            cur_coords = change_coords(start_coords, direction, 2)
        
        #write each branch of the individuals body, if it exists
        branch = body.core_v2.children[key]
        try:
            write_branch(branch.children[4], cur_coords, direction, grid)
        except KeyError:
            pass
    
    
    return grid


def calculate_difference(body1, body2):
    """score the difference:
        from nothing to something adds a distance of 1
        from hinge to brick or the other way around adds a distance of 0.5
        things can never go from or to core, this is always in the same place"""
    
    #get corresponding matrices
    if body2 != 'empty' and body1 != 'empty':
        matrix1 = get_filled_matrix(body1)
        matrix2 = get_filled_matrix(body2)
    elif body2 == 'empty' and body1 != 'empty':
        matrix1 = get_filled_matrix(body1)
        matrix2 = make_grid()
    elif body2 != 'empty' and body1 == 'empty':
        matrix1 = make_grid()
        matrix2 = get_filled_matrix(body2)
    else:
        print("Warning: Both bodies are empty, returning 0")
        return 0

    #for every cell, check if they are the same, or how different they are
    score = 0
    for row1, row2 in zip(matrix1, matrix2):
        for cell1, cell2 in zip(row1, row2):
            if (cell1 == 2 and cell2 == 3) or (cell1 == 3 and cell2 == 2):
                score += 0.5
            elif (cell1 == 2 and cell2 == 0) or (cell1 == 3 and cell2 == 0) or (cell1 == 0 and cell2 == 2) or (cell1 == 0 and cell2 == 3):
                score += 1
    
    return score


def get_size(body):
    """simply calculate the distance between just a core and the relevant body"""
    size = calculate_difference(body, 'empty')

    return size


"""
Functions that are specific to calculating heritability
"""


def add_sizes(df_population, df_genotype):
    """ Adds the sizes of the parents and offspring to the dataframe """

    #make empty columns
    df_population['size'] = np.NaN
    df_population['parent1_size'] = np.NaN
    df_population['parent2_size'] = np.NaN
    df_population['avg_parents_size'] = np.NaN

    #for every individual, calculate the size and add it to the dataframe, but only between the xlimits
    # for i, _ in df_population.iterrows():
    #     if i >= xlim[0] and i < xlim[1]:
    #         offspring_id = i
    #         body, body1, body2 = get_bodies_from_id(offspring_id, df_population, df_genotype)
    #         size = get_size(body)
    #         size1 = get_size(body1)
    #         size2 = get_size(body2)
    #         avg_size = (size1 + size2)/2

    #         df_population.loc[i, 'size'] = size
    #         df_population.loc[i, 'parent1_size'] = size1
    #         df_population.loc[i, 'parent2_size'] = size2
    #         df_population.loc[i, 'avg_parents_size'] = avg_size


    for i, _ in df_population.iterrows():
        offspring_id = i
        body, body1, body2 = get_bodies_from_id(offspring_id, df_population, df_genotype)
        size = get_size(body)
        size1 = get_size(body1)
        size2 = get_size(body2)
        avg_size = (size1 + size2)/2

        df_population.loc[i, 'size'] = size
        df_population.loc[i, 'parent1_size'] = size1
        df_population.loc[i, 'parent2_size'] = size2
        df_population.loc[i, 'avg_parents_size'] = avg_size


def calculate_heritability_individual_difference(offspring_id, df_population, df_genotype):
    if df_population['parent1_id'][offspring_id] == 0 or df_population['parent2_id'][offspring_id] == 0:
        return np.NaN
    #get the grids of each individual
    offspring_grid, parent1_grid, parent2_grid = get_bodies_from_id(offspring_id, df_population, df_genotype)
    #get the sizes of each individual
    offspring_size = df_population['size'][offspring_id]
    parent1_size = df_population['parent1_size'][offspring_id]
    parent2_size = df_population['parent2_size'][offspring_id]

    #calculate the heritability
    heritability = 1-((calculate_difference(offspring_grid, parent1_grid) + calculate_difference(offspring_grid, parent2_grid))/(parent1_size + (2 * offspring_size) + parent2_size + 0.000000000000000001))

    return heritability


def add_heritabilities_difference(df_population, df_genotype, add_sizes=False):
    #if the sizes need to be added, add them
    if add_sizes:
        add_sizes(df_population, df_genotype)
    
    #make name for column
    colname = "heritability_difference"

    #make empty column
    df_population[colname] = np.NaN

    #for every individual, calculate the heritability and add it to the dataframe
    for i, _ in df_population.iterrows():
        heritability = calculate_heritability_individual_difference(i, df_population, df_genotype)
        df_population.loc[i, colname] = heritability


def calculate_heritability_population_difference(generation, df_population, df_genotype, df_generation, add_values=False):
    """This function will calculate the heritability for every individual in the population"""
    #if the sizes need to be added, add them
    if add_values:
        add_heritabilities_difference(generation, df_population, df_genotype, df_generation, add_sizes=True)

    #copy dataframe
    df_populationc = df_population.copy()

    #use only data from the relevant generation
    df_populationc = get_generation_data(generation, df_generation, df_populationc)

    #heritability is the average of the heritabilities (take NaN values into account)
    heritability_difference = df_populationc['heritability_difference'].mean(skipna=True)

    return heritability_difference


def get_heritability_size(generation, df_generation, df_population):
    """get the actual heritability number"""
    #make name for column
    colname = "avg_parents_size"

    #copy both dataframes to keep the original
    df_generationc = df_generation.copy()
    df_populationc = df_population.copy()

    #drop all rows with NaN values
    df_populationc = df_populationc.dropna()
    
    #get relevant data
    df_gen = get_generation_data(generation, df_generationc, df_populationc)

    xdata = df_gen[colname]
    ydata = df_gen['size']

    #fit the data
    res = sm.OLS(ydata, xdata).fit()

    if res.params[0] > 1:
        print(f"Warning: Heritability in generation {generation} is higher than 1, adjusting...")
        return 1/res.params[0]
    else:
        return res.params[0]
    

"""
Visualization functions
"""

def plot_difference_heritability(generation, heritability, path, xlim=[0,1], ylim=[0,1]):
    """plots the heritability of the difference between individuals against a reference line with slope 1"""

    #make a figure
    fig, ax = plt.subplots()

    #plot reference line with slope 1
    x = np.linspace(-100, 100, 1000)
    y = x
    y2 = heritability * x
    ax.plot(x, y, color='green')
    ax.plot(x, y2, color='red')

    #set the limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    #make a title
    ax.set_title("Generation " + str(generation) + " Heritability: " + str(round(heritability, 2)))

    #make a legend
    ax.legend(["Perfect Heritability", "Population Heritability"])
    #save the figure
    plt.savefig(path + "/morphological/difference/generation_" + str(generation) + "_differenceheritability_scatter.png")


def plot_heritability_over_generations_difference(df_generation, df_population, df_genotype, path, xlim=[0,50], ylim=[0,1.2]):
    """plot the heritability over generations"""
    #make empty lists for the data
    generations = []
    slopes = []

    #retrieve the highest value in population_id
    max_gen = df_generation['generation_index'].max()

    #get the heritability for every generation
    for i in range(1, max_gen+1):
        slope = calculate_heritability_population_difference(i, df_population, df_genotype, df_generation)
        generations.append(i)
        slopes.append(slope)
        plot_difference_heritability(i, slope, path)

    #make an empty figure
    fig, ax = plt.subplots()

    #plot the data
    ax.plot(generations, slopes)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Heritability")
    ax.set_title("Heritability over generations for difference metric")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    #save the figure
    plt.savefig(path + f"/morphological/difference/difference_heritability_over_{xlim[1]}generations.png")


def plot_results_generation_size(generation, df_generation, df_population, slope, path, xlim=[0,22], ylim=[0,22]):
    """this will create and save a plot of the results of a specific dataframe and generation
    This does assume the corresponding column has already been created."""
    #make name for column
    colname = "avg_parents_size"

    df_gen = get_generation_data(generation, df_generation, df_population)

    xdata = df_gen[colname]
    ydata = df_gen['size']

    #make an empty figure
    fig, ax = plt.subplots()

    #plot the data
    ax.scatter(xdata, ydata)
    ax.set_xlabel("Average Parent Size")
    ax.set_ylabel('Size')
    ax.set_title("Generation " + str(generation) + " Heritability: " + str(round(slope, 2)))

    #plot the regression line
    x = np.linspace(-100, 100, 1000)
    y = slope * x
    ax.plot(x, y, color='red')

    #plot reference line with slope 1
    ax.plot(x, x, color='green')

    #add legend to the lines
    ax.legend(["Data", "Population Heritability", "Perfect Heritability"])

    #set the limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    #save the figure
    plt.savefig(path + "/morphological/size/generation_" + str(generation) + "_sizeheritability_scatter.png")


def plot_heritability_over_generations_size(df_generation, df_population, path, xlim=[0,50], ylim=[0,1.2]):
    """plot the heritability over generations"""
    #make empty lists for the data
    generations = []
    slopes = []

    #retrieve the highest value in population_id
    max_gen = df_generation['generation_index'].max()

    #get the heritability for every generation
    for i in range(1, max_gen+1):
        slope = get_heritability_size(i, df_generation, df_population)
        generations.append(i)
        slopes.append(slope)
        plot_results_generation_size(i, df_generation, df_population, slope, path)

    #make an empty figure
    fig, ax = plt.subplots()

    #plot the data
    ax.plot(generations, slopes)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Heritability")
    ax.set_title("Heritability over generations for size")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    #save the figure
    plt.savefig(path + f"/morphological/size/size_heritability_over_{xlim[1]}generations.png")


def show_matrix(bodyid, df_population, df_genotype, path, show=False):
    """ show the matrix in a figure """
    body, _, _ = get_bodies_from_id(bodyid, df_population, df_genotype)
    matrix = get_filled_matrix(body)
    plt.imshow(matrix, interpolation='nearest')
    
    if show:
        plt.show()

    #save the figure along with the cubes
    plt.imsave(f"{path}/morphological/bodies/individual_{bodyid}.png", matrix)
