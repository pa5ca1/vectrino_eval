import numpy as np
import pandas as pd
import os

# for saving files
from saving_data import saving_files
from analyse_data import plot_histogram_delta_t
from analyse_data import select_points_in_space
from analyse_data import plot_histogram_means_in_space
from analyse_data import plot_mean_in_space
from analyse_data import select_points_in_time


from load_experiment import load_exp
from load_experiment import cut_data

# For timing
from datetime import datetime

# May delete later

import matplotlib
import matplotlib.pyplot as plt
#font = {'family' : 'normal',
#        'weight' : 'bold',
#        'size'   : 22}
#matplotlib.rc('font', **font)
matplotlib.rcParams.update({'font.size': 22})

# Timer start
internal_timer_start = datetime.now()

# get current working directory
cwd = os.getcwd()

# Load column headers
file_headers = cwd + '/column_header_2.txt'

# Load headers
with open(file_headers) as f:
    headers = [line for line in f]
# Remove '\n'
headers = [head.replace('\n','') for head in headers]


## Load Experiment
exp_name = 'Pos_0_F45-16-W37-5'
df = load_exp(cwd,exp_name)
list_selected_points = [2,3]
delta_t = [10,20]
df_select = cut_data(df,list_selected_points,delta_t)



## General Plots
labels = pd.unique(df['label'])

plot_histogram_means_in_space(df_select,labels)

plot_histogram_delta_t(df)

plot_mean_in_space(df_select,n_datapoints,labels)

print(datetime.now() - internal_timer_start)
