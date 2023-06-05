import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# For timing
from datetime import datetime
startTime = datetime.now()

# get current working directory
cwd = os.getcwd()

# Load column headers
file_headers = cwd + '/column_header_2.txt'

# Load headers
with open(file_headers) as f:
    headers = [line for line in f]
# Remove '\n'
headers = [head.replace('\n','') for head in headers]

## Load files
load_all = True
sample_data_folder_path = cwd + '/../sample_data/'
# only use if one specific file should be used
file_name = 'VectrinoData.144.23.Vectrino Profiler.00006.dat'

if load_all:
    files = os.listdir(sample_data_folder_path)
    file_list = [file for file in files if file.endswith('.dat')]
else:
    file_list = [file_name]

# Define the line_names to extract
lines_to_extract = ['Profiles_Velocity_X','Profiles_Velocity_Y',
                    'Profiles_Velocity_Z1','Profiles_Velocity_Z2',
                    'Profiles_HostTime_start']

column_dict = {'Profiles_HostTime_start':1,
               'Profiles_Velocity_X':8,
               'Profiles_Velocity_Y':8,
               'Profiles_Velocity_Z1':8,
               'Profiles_Velocity_Z2':8}

main_column = [key for key, value in column_dict.items() for _ in range(value)]
sub_column = []
for key, value in column_dict.items():
    sub_column.extend(list(range(1, value + 1)))

# How to name the columns in pandas dataframe
column_headers = [np.array(main_column),np.array(sub_column)]

# Define saving format
save_format = 'csv'

for file_sample_name in file_list:
    # Choosing the correct file
    file_sample = cwd + '/../sample_data/' + file_sample_name
    # We delete the .dat at the end and put everything else together
    output_filename = os.path.splitext(file_sample_name)[0]

    # extract lines from sample file
    with open(file_sample) as f:
        lines = [line for line in f]

    #df = pd.DataFrame(columns=headers)
    mem_list = []
    big_list = []

    for line in lines:
    # Each line consits of two parts: First a name and second the value, seperated by colon
        parts = line.split(":")
        column_name = parts[0].strip()
        column_name = column_name.split()[0]

        
        # Do not use the values with 'VelocityHeader' in name
        if column_name in lines_to_extract:
        
            if '[' not in parts[1]:
                # unix time
                after_colon = float(parts[1].strip().replace(",", "."))
                mem_list.append(after_colon)
            else:
                # experimental values
                after_colon = [x.replace(',', '.') for x in parts[1].split()][1:-1]
                # Check if we read a status which is only an integer values or not
                if len(after_colon) > 1:
                    # Not a status
                    after_colon = [float(x) for x in after_colon]
                    mem_list = mem_list + after_colon
                else:
                    # Status
                    after_colon = after_colon[0]
                    mem_list.append(after_colon)
            # We now have arrays in after colon
            

        if column_name == 'Profiles_AveragedPingPairs':
            big_list.append(mem_list)
            mem_list = []


    df = pd.DataFrame(big_list,columns=column_headers)
    #df.to_csv(output_destination)

    # Saving results
    output_destination = sample_data_folder_path + save_format + '/' + output_filename
    if save_format == 'csv':
        output_destination = output_destination + '.csv'
        df.to_csv(output_destination)
    elif save_format == 'h5':
        output_destination = output_destination + '.h5'
        df.to_hdf(output_destination,key='df')
    else:
        None


    # Analysing data
    # Select only feasible points under sensor
    list_selected_points = [2,3,4]
    select = df.columns.get_level_values(1).isin(list_selected_points)
    df_analyse = df.loc[:, select]
    x_mean_velocity = df_analyse.Profiles_Velocity_X.mean(axis=1).to_numpy()
    y_mean_velocity = df_analyse.Profiles_Velocity_Y.mean(axis=1).to_numpy()
    z1_mean_velocity = df_analyse.Profiles_Velocity_Z1.mean(axis=1).to_numpy()
    z2_mean_velocity = df_analyse.Profiles_Velocity_Z2.mean(axis=1).to_numpy()

    plt.hist(x_mean_velocity)
    plt.show()


print(datetime.now() - startTime)
