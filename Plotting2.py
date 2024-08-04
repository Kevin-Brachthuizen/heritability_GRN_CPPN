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

generation_df_cppn = pd.read_sql_query("SELECT * from generation", con)
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


bh.plot_results_generation(1, generation_df_cppn, individual_df_cppn, bh.get_heritability(1, generation_df_cppn, individual_df_cppn, "fitness"), 'plots/poster/CPPN')
bh.plot_results_generation(1, generation_df_grn, individual_df_grn, bh.get_heritability(1, generation_df_grn, individual_df_grn, "fitness"), 'plots/poster/GRN')

    



