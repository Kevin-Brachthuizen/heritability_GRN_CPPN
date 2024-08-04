import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


def get_parent_ids(ind_id, df):
    "Get the parent ids from an individuals id"
    #get parent ids
    parent1_id = df['parent1_id'][ind_id]
    parent2_id = df['parent2_id'][ind_id]

    return parent1_id, parent2_id


def get_avg_parentscore(ind_id, df_original, metric:str="fitness"):
    """get the average scores of the parents given their id and a dataframe"""
    #copy dataframe to keep original
    df = df_original.copy()
    #get parent ids
    parent1_id, parent2_id = get_parent_ids(ind_id, df)

    #retrieve individual scores of the parents
    #note that we only take the first instance of each list, since every individual will be the same but they might survive more than one generation
    #we also do not get the avg score for first generation individuals
    if parent1_id != -1 or parent2_id != -1:
        parent1 = df.loc[df["genotype_id"] == parent1_id]
        parent1 = parent1.reset_index(drop=True)
        parent1_score = parent1[metric][0]

        parent2 = df.loc[df["genotype_id"] == parent2_id]
        parent2 = parent2.reset_index(drop=True)
        parent2_score = parent2[metric][0]

        #calculate average
        avg_score = (parent1_score + parent2_score)/2
        return avg_score
    else:
        return np.NaN


def add_avg_parentscores(df, metric:str="fitness"):
    """add the average score of the parents of some metric to the dataframe"""
    #make name for column
    colname = "avg_parentscore_" + metric

    #make empty column
    df[colname] = np.NaN

    #get average score for every individual and add to dataframe
    for i, _ in df.iterrows():
        avg = get_avg_parentscore(i, df, metric)
        df.loc[i, colname] = avg


def get_generation_data(generation, df_generation, df_population):
    """get the data for a specific generation"""
    #copy both dataframes to keep the original
    df_generationc = df_generation.copy()
    df_populationc = df_population.copy()

    #drop all rows with NaN values
    df_populationc = df_populationc.dropna()
    
    #retrieve the population id's from df_generation for which the generation index is the same as generation
    population_ids = df_generationc.loc[df_generationc['generation_index'] == generation]['population_id']

    #retrieve for each experiment the separate population ids


    #get the data for the specific generation
    df_gen = df_populationc.loc[df_populationc['population_id'].isin(population_ids)]

    return df_gen

def get_experiment_data(df_generation, df_population):
    #copy both dataframes to keep the original
    df_generationc = df_generation.copy()
    df_populationc = df_population.copy()

    #drop all rows with NaN values
    df_populationc = df_populationc.dropna()

    #get for every experiment id a separate dataframe
    experiment1 = df_generationc.loc[df_generationc['experiment_id'] == 1]
    experiment2 = df_generationc.loc[df_generationc['experiment_id'] == 2]
    experiment3 = df_generationc.loc[df_generationc['experiment_id'] == 3]
    experiment4 = df_generationc.loc[df_generationc['experiment_id'] == 4]
    experiment5 = df_generationc.loc[df_generationc['experiment_id'] == 5]
    experiment6 = df_generationc.loc[df_generationc['experiment_id'] == 6]
    experiment7 = df_generationc.loc[df_generationc['experiment_id'] == 7]
    experiment8 = df_generationc.loc[df_generationc['experiment_id'] == 8]
    experiment9 = df_generationc.loc[df_generationc['experiment_id'] == 9]
    experiment10 = df_generationc.loc[df_generationc['experiment_id'] == 10]

    return [experiment1, experiment2, experiment3, experiment4, experiment5, experiment6, experiment7, experiment8, experiment9, experiment10]


def get_heritability(generation, df_generation, df_population, metric:str="fitness"):
    """get the actual heritability number"""
    #make name for column
    colname = "avg_parentscore_" + metric

    #copy both dataframes to keep the original
    df_generationc = df_generation.copy()
    df_populationc = df_population.copy()

    #drop all rows with NaN values
    df_populationc = df_populationc.dropna()
    
    #get relevant data
    df_gen = get_generation_data(generation, df_generationc, df_populationc)

    xdata = df_gen[colname]
    ydata = df_gen[metric]

    #fit the data
    res = sm.OLS(ydata, xdata).fit()

    if res.params[0] > 1:
        print(f"Warning: Heritability in generation {generation} is higher than 1, adjusting...")
        return 1/res.params[0]
    else:
        return res.params[0]


def plot_results_generation(generation, df_generation, df_population, slope, path, metric:str="fitness", xlim=[0,5], ylim=[0,5]):
    """this will create and save a plot of the results of a specific dataframe and generation
    This does assume the corresponding column has already been created."""
    #make name for column

    colname = "avg_parentscore_" + metric

    df_gen = get_generation_data(generation, df_generation, df_population)

    xdata = df_gen[colname]
    ydata = df_gen[metric]

    #make an empty figure
    fig, ax = plt.subplots()

    #plot the data
    ax.scatter(xdata, ydata)
    if metric == "fitness":
        ax.set_xlabel(f"Average Parent speed")
        ax.set_ylabel('Speed')
    else:
        ax.set_xlabel(f"Average Parent {metric}")
        ax.set_ylabel(metric)
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
    plt.savefig(path + "generation_" + str(generation) + "_" + metric + "heritability_scatter.png")


def plot_heritability_over_generations(df_generation, df_population, path, metric:str="fitness", xlim=[0,50], ylim=[0,1.2]):
    """plot the heritability over generations"""
    #make empty lists for the data
    generations = []
    slopes = []

    if metric == "fitness":
        xlimc = [0,5]
        ylimc = [0,5]
    elif metric == "balance":
        xlimc = [0.8,1]
        ylimc = [0.8,1]
    #retrieve the highest value in population_id
    max_gen = df_generation['generation_index'].max()

    #get the heritability for every generation
    for i in range(1, max_gen+1):
        slope = get_heritability(i, df_generation, df_population, metric)
        generations.append(i)
        slopes.append(slope)
        plot_results_generation(i, df_generation, df_population, slope, path, metric, xlimc, ylimc)

    #make an empty figure
    fig, ax = plt.subplots()

    #plot the data
    ax.plot(generations, slopes)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Heritability")
    ax.set_title("Heritability over generations for " + metric)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    #save the figure
    plt.savefig(path + "/behavioural/" + metric + "/heritability_over_generations.png")