import sqlite3
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

from behavioural_heritability_functions import add_avg_parentscores, plot_results_generation, plot_heritability_over_generations

database_name = 'experiments_data/GRN.sqlite'
if os.path.exists(database_name) == True:
    con = sqlite3.connect(database_name)
else:
    print(os.path.exists(database_name))
    raise KeyError("No such file or directory (you might be in the wrong directory to begin with)")

generation_df = pd.read_sql_query("SELECT * from generation", con)
genotype_df = pd.read_sql_query("SELECT * from genotype", con)
individual_df = pd.read_sql_query("SELECT * from individual", con)


print('Plotting heritability over generations')
plot_heritability_over_generations(generation_df, individual_df, 'plots/GRN/', xlim=[0, 100], ylim=[0, 1.2])
plot_heritability_over_generations(generation_df, individual_df, 'plots/GRN/', metric="balance", xlim=[0, 100], ylim=[0, 1.2])

# database_name = 'experiments_data/CPPN.sqlite'
# if os.path.exists(database_name) == True:
#     con = sqlite3.connect(database_name)
# else:
#     print(os.path.exists(database_name))
#     raise KeyError("No such file or directory (you might be in the wrong directory to begin with)")

# generation_df = pd.read_sql_query("SELECT * from generation", con)
# genotype_df = pd.read_sql_query("SELECT * from genotype", con)
# individual_df = pd.read_sql_query("SELECT * from individual", con)
 
# print('adding scores')
# add_avg_parentscores(individual_df, metric="fitness")
# add_avg_parentscores(individual_df, metric="balance")

# print('Plotting heritability over generations')
# plot_heritability_over_generations(generation_df, individual_df, 'plots/CPPN/', xlim=[0, 100], ylim=[0, 1.2])
# plot_heritability_over_generations(generation_df, individual_df, 'plots/CPPN/', metric="balance", xlim=[0, 100], ylim=[0, 1.2])



