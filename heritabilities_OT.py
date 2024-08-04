import sqlite3
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import multineat

from scipy.stats import shapiro

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

#this works for fitness and balance. For size and distance the slope calculation in the for loop needs to be changed
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
        slope = bh.get_heritability(i, experiment_generation, individual_df_cppn, metric) #metric
        # slope = mh.calculate_heritability_population_difference(i, individual_df_cppn, genotype_df_cppn, experiment_generation) #distance
        # slope = mh.get_heritability_size(i, experiment_generation, individual_df_cppn) #size
        slopes_cppn = np.append(slopes_cppn, slope)
    #add the slopes to heritabilitylists as an array
    heritabilitylists_cppn = np.vstack((heritabilitylists_cppn, slopes_cppn))

for experiment_generation in experimentlist_grn:
    #slopes is an empty array
    slopes_grn = np.array([])
    for i in range(1, max_gen+1):
        slope = bh.get_heritability(i, experiment_generation, individual_df_grn, metric)
        # slope = mh.calculate_heritability_population_difference(i, individual_df_grn, genotype_df_grn, experiment_generation)
        # slope = mh.get_heritability_size(i, experiment_generation, individual_df_grn)
        slopes_grn = np.append(slopes_grn, slope)
    #add the slopes to heritabilitylists as an array
    heritabilitylists_grn = np.vstack((heritabilitylists_grn, slopes_grn))

#get the median of generation over all experiments
medians_cppn = np.median(heritabilitylists_cppn, axis=0)
medians_grn = np.median(heritabilitylists_grn, axis=0)

#get the max values for each generation over all experiments
maxs_cppn = np.max(heritabilitylists_cppn, axis=0)
maxs_grn = np.max(heritabilitylists_grn, axis=0)

#get the min values for each generation over all experiments
mins_cppn = np.min(heritabilitylists_cppn, axis=0)
mins_grn = np.min(heritabilitylists_grn, axis=0)

#make an empty figure
fig, ax = plt.subplots()

#plot the data
ax.plot(generations, medians_cppn, label="CPPN")
ax.fill_between(generations, mins_cppn, maxs_cppn, alpha=0.2)

ax.plot(generations, medians_grn, label="GRN")
ax.fill_between(generations, mins_grn, maxs_grn, alpha=0.2)
ax.set_xlabel("Generation")
ax.set_ylabel("Heritability")
ax.set_title("Heritability over generations for speed")
ax.set_xlim(0,50)
ax.set_ylim(0,1.1)
ax.legend(loc='lower right')

#save the figure
plt.savefig('plots/tests' + f"/{metric}_heritability_over_generations.png")

