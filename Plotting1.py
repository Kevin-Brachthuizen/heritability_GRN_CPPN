import sqlite3
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

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
        # slope = bh.get_heritability(i, experiment_generation, individual_df_cppn, metric) #metric
        # slope = mh.get_heritability_size(i, experiment_generation, individual_df_cppn) #size
        slope = mh.calculate_heritability_population_difference(i, individual_df_cppn, genotype_df_cppn, experiment_generation) #distance
        slopes_cppn = np.append(slopes_cppn, slope)
    #add the slopes to heritabilitylists as an array
    heritabilitylists_cppn = np.vstack((heritabilitylists_cppn, slopes_cppn))

for experiment_generation in experimentlist_grn:
    #slopes is an empty array
    slopes_grn = np.array([])
    for i in range(1, max_gen+1):
        # slope = bh.get_heritability(i, experiment_generation, individual_df_grn, metric)
        # slope = mh.get_heritability_size(i, experiment_generation, individual_df_grn)
        slope = mh.calculate_heritability_population_difference(i, individual_df_grn, genotype_df_grn, experiment_generation)
        slopes_grn = np.append(slopes_grn, slope)
    #add the slopes to heritabilitylists as an array
    heritabilitylists_grn = np.vstack((heritabilitylists_grn, slopes_grn))

#get the median of generation over all experiments
medians_cppn = np.median(heritabilitylists_cppn, axis=0)
mins_cppn = np.min(heritabilitylists_cppn, axis=0)
maxs_cppn = np.max(heritabilitylists_cppn, axis=0)
medians_grn = np.median(heritabilitylists_grn, axis=0)
mins_grn = np.min(heritabilitylists_grn, axis=0)
maxs_grn = np.max(heritabilitylists_grn, axis=0)

median_metric_cppn = []
min_metric_cppn = []
max_metric_cppn = []
median_metric_grn = []
min_metric_grn = []
max_metric_grn = []

for i in range(1, max_gen+1):
    data_grn = bh.get_generation_data(i, generation_df_grn, individual_df_grn)
    data_cppn = bh.get_generation_data(i, generation_df_cppn, individual_df_cppn)
    Q1, _, Q3 = np.quantile(data_grn[metric], [0.25, 0.5, 0.75], method='nearest')
    median_metric_grn.append(data_grn[metric].median())
    min_metric_grn.append(Q1)
    max_metric_grn.append(Q3)
    median_metric_cppn.append(data_cppn[metric].median())
    Q1, _, Q3 = np.quantile(data_cppn[metric], [0.25, 0.5, 0.75], method='nearest')
    min_metric_cppn.append(Q1)
    max_metric_cppn.append(Q3)

fig, ax = plt.subplots()
ax2 = ax.twinx()
line = plt.Line2D((0,1),(0,1), color='black', linestyle='solid')
dash = plt.Line2D((0,1),(0,1), color='black', linestyle='dashed')

#dashed line
ax.plot(generations, medians_cppn, label='CPPN', color='C0')
# ax.fill_between(generations, mins_cppn, maxs_cppn, alpha=0.2)
ax.plot(generations, medians_grn, label='GRN', color='orange')
# ax.fill_between(generations, mins_grn, maxs_grn, alpha=0.2)
ax2.plot(generations, median_metric_cppn, label='CPPN', linestyle='dashed', color='C0')
ax2.fill_between(generations, min_metric_cppn, max_metric_cppn, alpha=0.2)
ax2.plot(generations, median_metric_grn, label='GRN', linestyle='dashed', color='orange')
ax2.fill_between(generations, min_metric_grn, max_metric_grn, alpha=0.2)
ax.set_xlabel('Generation')
ax.set_ylabel('Median Heritability')
ax.set_ylim(0,1.1)
ax2.set_ylabel('Median speed')
ax.legend(loc='lower right')
ax2.legend([line, dash], ['Median Heritability', "Median speed"], loc='lower center')
ax.set_title(f'Progression of heritability vs speed over generations')

#save the plot
fig.savefig(f"plots/poster/differenceheritability_vs_speed.png")









    



