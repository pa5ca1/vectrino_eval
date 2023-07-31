import numpy as np
import pandas as pd

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


def calculate_turbulence_intensity(df_select,df_mean):

    I = []
    labels = df_select.label.unique()
    for label in labels:
        u_x_mean = df_mean[df_mean.label == label].u_x.iloc[0]
        u_y_mean = df_mean[df_mean.label == label].u_y.iloc[0]
        u_z1_mean = df_mean[df_mean.label == label].u_z1.iloc[0]
        u_z2_mean = df_mean[df_mean.label == label].u_z2.iloc[0]

        u_x = df_select[df_select.label == label].Profiles_Velocity_X.to_numpy().reshape(-1)
        u_dash_sq_mean_x = np.sqrt(np.mean((u_x-u_x_mean)**2))

        u_y = df_select[df_select.label == label].Profiles_Velocity_Y.to_numpy().reshape(-1)
        u_dash_sq_mean_y = np.sqrt(np.mean((u_y-u_y_mean)**2))

        u_z1 = df_select[df_select.label == label].Profiles_Velocity_Z1.to_numpy().reshape(-1)
        u_dash_sq_mean_z1 = np.sqrt(np.mean((u_z1-u_z1_mean)**2))

        u_z2 = df_select[df_select.label == label].Profiles_Velocity_Z2.to_numpy().reshape(-1)
        u_dash_sq_mean_z2 = np.mean((u_z2-u_z2_mean)**2)   

        u_dash = np.sqrt((u_dash_sq_mean_x + u_dash_sq_mean_y + u_dash_sq_mean_z1)/3)
        u_hat = np.sqrt(u_x_mean**2 + u_y_mean**2 + u_z1_mean**2)
        
        I.append(u_dash/u_hat)
    df_mean['turb_int'] = I


def calculate_mean_per_measurement(df):

    u_x = []
    u_y = []
    u_z1 = []
    u_z2 = []
    std_vel = []
    t_intervall = []
    labels = df.label.unique()


    for label in labels:
        mean_values = df[df.label == label].mean(numeric_only=True)
        std_values = df[df.label == label].std(numeric_only=True)
        u_x.append(mean_values.Profiles_Velocity_X.mean())
        u_y.append(mean_values.Profiles_Velocity_Y.mean())
        u_z1.append(mean_values.Profiles_Velocity_Z1.mean())
        u_z2.append(mean_values.Profiles_Velocity_Z2.mean())
        t1 = df[df.label == label].Profiles_HostTime_start.iloc[-1]
        t2 = df[df.label == label].Profiles_HostTime_start.iloc[0]
        t_intervall.append((t2,t1))
        x_std = std_values.Profiles_Velocity_X.std()
        y_std = std_values.Profiles_Velocity_Y.std()
        z1_std = std_values.Profiles_Velocity_Z1.std()
        z2_std = std_values.Profiles_Velocity_Z2.std()
        std_vel.append([x_std,y_std,z1_std,z2_std])

    
    mean_df = pd.DataFrame()
    mean_df['u_x'] = u_x
    mean_df['u_y'] = u_y
    mean_df['u_z1'] = u_z1
    mean_df['u_z2'] = u_z2
    std_vel = np.array(std_vel)
    mean_df['x_std'] = std_vel[:,0]
    mean_df['y_std'] = std_vel[:,1]
    mean_df['z1_std'] = std_vel[:,2]
    mean_df['z2_std'] = std_vel[:,3]
    mean_df['time_intervall'] = t_intervall
    mean_df['label'] = labels

    return mean_df

