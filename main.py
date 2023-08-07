import numpy as np
import pandas as pd
import os

# For timing
from datetime import datetime
from src.experiment import Experiment

from src.plot_experiment import plot_turb_int


import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 22})

# Timer start
internal_timer_start = datetime.now()




cwd = os.getcwd()
result_data_path =  cwd + '/sample_data/'
files = os.listdir(result_data_path)
all_exp = [file for file in files if not file.endswith('.DS_Store')]

for exp_name in all_exp:

    pos_name = 'pos' + exp_name.split('_')[1]

    exp = Experiment(exp_name,pos_name)
    exp.save_data('turbulence_intensity')
    plot_turb_int()

print(datetime.now() - internal_timer_start)
