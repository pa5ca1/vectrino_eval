
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_histogram_delta_t(df):
    plt.figure(1)
    df.iloc[:,0].diff().hist()
    plt.yscale('log')
    plt.title('Delta t')
    plt.xlabel('Delta t [s]')
    plt.ylabel('log. number of value')
    plt.show()


def plot_mean_in_space(df_select,n_datapoints,labels):
    plt.figure()
    mean_matrix = np.zeros(n_datapoints)

    for i,label in enumerate(labels):
        mean_matrix[i] = df_select[df_select['label']==label].Profiles_Velocity_X.to_numpy().reshape(-1).mean()

    mean_matrix = mean_matrix.reshape(11,7)
    plt.imshow(mean_matrix.transpose(),cmap=plt.cm.bwr)
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.title('Mean per point in space')
    plt.colorbar()

def plot_turbulence_intensity_heatmap(df,shape):
    
    plot_matrix = df.turb_int.to_numpy().reshape(shape)
    plt.figure()
    plt.imshow(plot_matrix.transpose(),cmap=plt.cm.bwr)
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.title('Turbulence Intensity')
    plt.colorbar()
    
    

def plot_histogram_means_in_space(df_select,labels):

    df_analyse = pd.DataFrame(columns=['label','mean_val'])
    df_analyse['label'] = df_select.label
    df_analyse['mean_val'] = df_select.Profiles_Velocity_X.mean(axis=1)
    fig, axs = plt.subplots(11, 7, sharex = False, sharey = False)
    fig.set_size_inches(26, 18)
    fig.suptitle('mean')
    fig.set_dpi(100)
    fig.tight_layout()
    
    for i,ax in enumerate(axs.reshape(-1)):
        data = df_analyse[df_analyse['label']==labels[i]]['mean_val'].to_numpy()
        ax.hist(data,density=False)