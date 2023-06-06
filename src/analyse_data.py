
import matplotlib.pyplot as plt


def plot_histogram_delta_t(df):
    plt.figure(1)
    df.iloc[:,0].diff().hist()
    plt.yscale('log')
    plt.title('Delta t')
    plt.xlabel('Delta t [s]')
    plt.ylabel('log. number of value')
    plt.show()

def adding_meassurement_label(df,n_datapoints):
    '''Adds a label for each meassurement if a) a sample is taken or b) the traverse is moving.
    The decision is based only on the time stamp.
    '''
    # Time per datapoint
    end_time = df.Profiles_HostTime_start.max()
    start_time = df.Profiles_HostTime_start.min()
    t_per_datapoint = (end_time-start_time)/n_datapoints
    list_periods = [start_time+i*t_per_datapoint for i in range(n_datapoints+1)]

    df['label'] = df['Profiles_HostTime_start'].iloc[0].apply(get_label,l=list_periods)


def get_label(timestamp,**kwargs):
    label_list = kwargs['l']
    for i in range(len(label_list)-1):
        if label_list[i] <= timestamp < label_list[i+1]:
            return chr(65 + i)  # Convert to corresponding label ('A', 'B', 'C', ...)
    return None