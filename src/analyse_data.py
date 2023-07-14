
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
    mean_matrix = np.zeros(n_datapoints)

    for i,label in enumerate(labels):
        mean_matrix[i] = df_select[df_select['label']==label].Profiles_Velocity_X.to_numpy().reshape(-1).mean()

    mean_matrix = mean_matrix.reshape(11,7)
    plt.imshow(mean_matrix.transpose())
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.title('Mean per point in space')
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
    

def adding_measurement_label(df,n_datapoints):
    '''Adds a label for each meassurement if a) a sample is taken or b) the traverse is moving.
    The decision is based only on the time stamp.
    '''
    # Time per datapoint
    end_time = df.Profiles_HostTime_start.max()
    start_time = df.Profiles_HostTime_start.min()
    t_per_datapoint = (end_time-start_time)/n_datapoints
    # In list_periods must be the start and end times of each sample point
    # sampling time.
    # Example: [t_0, t_1, t_2, t_3, ..., t_n] with t_1 < t_2 < t_3 ... t_n
    # and t_n = end_time
    list_periods = np.array([start_time+i*t_per_datapoint for i in range(n_datapoints+1)])
    # We use .squezze to change from pandas dataframe to timeseries
    # because 'get_label' needs a timeseries
    df['label'] = df['Profiles_HostTime_start'].squeeze().apply(get_label,l=list_periods)

    return df

def select_points_in_space(df,list_selected_points):

    select = df.columns.get_level_values(1).isin(list_selected_points)
    # We want to keep the column 'label'
    select[-1] = True
    # We want to keep timestamps
    #select[0] = True
    df_select = df.loc[:, select]
    # We want to keep timestamps
    df_select['Profiles_HostTime_start'] = df.Profiles_HostTime_start.to_numpy()
    return df_select


def select_points_in_time(df,delta_t):
    '''This function drops the first and last recorded time points
    of the measurement and returns the new dataframe with only the 'middle' 
    time points.
    t_0,t_1   : is dropped
    t_1,t_2 : stays
    i_2,t_3 : is dropped
    '''
    # Start from the bottom
    labels = np.flip(df.label.unique())
    
    for label in labels:
        t_0 = df[df.label == label].Profiles_HostTime_start.to_numpy()[0]
        t_3 =df[df.label == label].Profiles_HostTime_start.to_numpy()[-1]
        t_1 = t_0 + delta_t[0]
        t_2 = t_3 - delta_t[1]
        i_0 = df[(df.Profiles_HostTime_start>=t_0) & (df.Profiles_HostTime_start<t_1)].index
        i_1 = df[(df.Profiles_HostTime_start>t_2) & (df.Profiles_HostTime_start<=t_3)].index
        df = df.drop(i_1).drop(i_0)
        #df_select = df_select[df.label == label].drop(range(0,i_0))
    return df

def get_label(timestamp,**kwargs):
    ''' Labeling each '''
    
    label_list = kwargs.get('l', None)

    for i in range(len(label_list)-1):
        if label_list[i] <= timestamp <= label_list[i+1]:
            return chr(65 + i)  # Convert to corresponding label ('A', 'B', 'C', ...)
    return 'Test'


def calculate_mean_per_measurement(df):
    return None