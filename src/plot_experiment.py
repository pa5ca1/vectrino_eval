import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_turb_int():

    cwd = os.getcwd()
    result_data_path =  cwd + '/results_data/'
    files = os.listdir(result_data_path)
    list_of_all_files = [file for file in files if file.endswith('.npy')]

    for file in list_of_all_files:

        turb_int = np.load(result_data_path + file)

        plt.imshow(turb_int.transpose(),
                    cmap=plt.cm.bwr,origin='lower')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(file,fontsize=12)
        plt.colorbar()
        plot_save_path = cwd + '/figures/'
        plt.savefig(plot_save_path + file + '.png')
        plt.close()
