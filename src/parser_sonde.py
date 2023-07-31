import numpy as np
import pandas as pd
import os


from plots import plot_turbulence_intensity_heatmap


from load_experiment import load_exp
from load_experiment import cut_data

from analyse_data import calculate_mean_per_measurement
from analyse_data import calculate_turbulence_intensity

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
#exp_name = 'Pos_0_F45-16-W37-5'
exp_name = 'Pos_1_F49-07_W57-1'

df = load_exp(cwd,exp_name)
list_selected_points = [3]
delta_t = [10,20]
df_select = cut_data(df,list_selected_points,delta_t)
df_mean = calculate_mean_per_measurement(df_select)
calculate_turbulence_intensity(df_select,df_mean)


## General Plots
labels = pd.unique(df['label'])

plot_turbulence_intensity_heatmap(df_mean,(11,7))


print(datetime.now() - internal_timer_start)
