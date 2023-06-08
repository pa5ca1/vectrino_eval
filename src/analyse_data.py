
import matplotlib.pyplot as plt
import numpy as np


def plot_histogram_delta_t(df):
    plt.figure(1)
    df.iloc[:,0].diff().hist()
    plt.yscale('log')
    plt.title('Delta t')
    plt.xlabel('Delta t [s]')
    plt.ylabel('log. number of value')
    plt.show()

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


def get_label(timestamp,**kwargs):
    ''' Labeling each '''
    
    label_list = kwargs.get('l', None)

    for i in range(len(label_list)-1):
        if label_list[i] <= timestamp <= label_list[i+1]:
            return chr(65 + i)  # Convert to corresponding label ('A', 'B', 'C', ...)
    return 'Test'


def calculate_mean_per_measurement(df):
    return None