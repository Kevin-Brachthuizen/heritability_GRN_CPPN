import sqlite3
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import multineat

from scipy.stats import shapiro, ttest_ind, f_oneway, wilcoxon

import morphological_heritability_functions as mh
import behavioural_heritability_functions as bh

database_name = 'experiments_data/CPPN.sqlite'
if os.path.exists(database_name) == True:
    con = sqlite3.connect(database_name)
else:
    print(os.path.exists(database_name))
    raise KeyError("No such file or directory (you might be in the wrong directory to begin with)")

generation_df_cppn= pd.read_sql_query("SELECT * from generation", con)
genotype_df_cppn = pd.read_sql_query("SELECT * from genotype", con)
individual_df_cppn = pd.read_sql_query("SELECT * from individual", con)


database_name = 'experiments_data/GRN.sqlite'
if os.path.exists(database_name) == True:
    con = sqlite3.connect(database_name)
else:
    print(os.path.exists(database_name))
    raise KeyError("No such file or directory (you might be in the wrong directory to begin with)")

generation_df_grn = pd.read_sql_query("SELECT * from generation", con)
genotype_df_grn = pd.read_sql_query("SELECT * from genotype", con)
individual_df_grn = pd.read_sql_query("SELECT * from individual", con)

#add 1 to every size
individual_df_cppn['size'] = individual_df_cppn['size'] + 1
individual_df_grn['size'] = individual_df_grn['size'] + 1

#this works for fitness and balance. For size and distance the slope calculation in the for loop needs to be changed.
metric = 'fitness'
experimentlist_cppn = bh.get_experiment_data(generation_df_cppn, individual_df_cppn)
experimentlist_grn = bh.get_experiment_data(generation_df_grn, individual_df_grn)


#make a variable that is a list of lists
heritabilitylists_grn = np.ndarray(shape=(0,50))
heritabilitylists_cppn = np.ndarray(shape=(0,50))

#retrieve the highest value in population_id
max_gen = generation_df_grn['generation_index'].max()
generations = range(1, max_gen+1)

#get the heritability for every generation
for experiment_generation in experimentlist_cppn:
    #slopes is an empty array
    slopes_cppn = np.array([])
    for i in range(1, max_gen+1):
        slope = bh.get_heritability(i, experiment_generation, individual_df_cppn, metric)
        data = bh.get_generation_data(i, experiment_generation, individual_df_cppn)
        median_fitness = data['fitness'].mean()
        # slope = mh.get_heritability_size(i, experiment_generation, individual_df_cppn) #size
        # slope = mh.calculate_heritability_population_difference(i, individual_df_cppn, genotype_df_cppn, experiment_generation) #distance
        slopes_cppn = np.append(slopes_cppn, median_fitness)
    #add the slopes to heritabilitylists as an array
    heritabilitylists_cppn = np.vstack((heritabilitylists_cppn, slopes_cppn))

for experiment_generation in experimentlist_grn:
    #slopes is an empty array
    slopes_grn = np.array([])
    for i in range(1, max_gen+1):
        # slope = bh.get_heritability(i, experiment_generation, individual_df_grn, metric)
        data = bh.get_generation_data(i, experiment_generation, individual_df_grn)
        median_fitness = data['fitness'].mean()
        # slope = mh.get_heritability_size(i, experiment_generation, individual_df_grn)
        # slope = mh.calculate_heritability_population_difference(i, individual_df_grn, genotype_df_grn, experiment_generation)
        slopes_grn = np.append(slopes_grn, median_fitness)
    #add the slopes to heritabilitylists as an array
    heritabilitylists_grn = np.vstack((heritabilitylists_grn, slopes_grn))
#get a vertical slice of data
heritabilitylists_cppn = heritabilitylists_cppn.T
heritabilitylists_grn = heritabilitylists_grn.T
gen = 0

print(f"{metric}")
for gen in range(len(heritabilitylists_cppn)):
    generation_cppn = heritabilitylists_cppn[gen]
    generation_grn = heritabilitylists_grn[gen]
    print("________________________________________________________")
    print(f"Gen {gen+1}")
    print(f"CPPN shapiro: {shapiro(generation_cppn)}")
    print(f"GRN shapiro: {shapiro(generation_grn)}")

    print(f"NoVariance ttest: {ttest_ind(generation_cppn, generation_grn, equal_var=True)}")
    print(f"Variance ttest: {ttest_ind(generation_cppn, generation_grn, equal_var=False)}")
    print(f"Wilcoxon: {wilcoxon(generation_cppn, generation_grn)}")
