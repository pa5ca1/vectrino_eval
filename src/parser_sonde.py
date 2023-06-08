import numpy as np
import pandas as pd
import os

# for saving files
from saving_data import saving_files
from analyse_data import plot_histogram_delta_t
from analyse_data import adding_measurement_label

# For timing
from datetime import datetime


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
    list_of_all_files = [file for file in files if file.endswith('.dat')]
else:
    list_of_all_files = [file_name]
# Sorting the files alphabetically
list_of_all_files.sort()
# Number of all datapoints
n_datapoints = len(list_of_all_files)


## Define saving format
save_format = 'csv'
bool_save = False

## Define the line_names to extract
lines_to_extract = ['Profiles_Velocity_X','Profiles_Velocity_Y',
                    'Profiles_Velocity_Z1','Profiles_Velocity_Z2',
                    'Profiles_HostTime_start']
# Number of sample points per direction
n_sample_depth = 7

column_dict = {'Profiles_HostTime_start':1,
               'Profiles_Velocity_X':n_sample_depth,
               'Profiles_Velocity_Y':n_sample_depth,
               'Profiles_Velocity_Z1':n_sample_depth,
               'Profiles_Velocity_Z2':n_sample_depth}

# How to name the columns in pandas dataframe
main_column = [key for key, value in column_dict.items() for _ in range(value)]
sub_column = []
for key, value in column_dict.items():
    sub_column.extend(list(range(1, value + 1)))
column_headers = [np.array(main_column),np.array(sub_column)]

exp_list = []

for i,file_sample_name in enumerate(list_of_all_files):
    print('Sampling: ' + str(i+1) +' of ' + str(n_datapoints) + ' | Name: ' + file_sample_name)
    # Choosing the correct file
    file_sample = cwd + '/../sample_data/' + file_sample_name
    # We delete the string '.dat' at the end and put everything else together
    output_filename = os.path.splitext(file_sample_name)[0]

    # extract lines from sample file
    with open(file_sample) as f:
        lines = [line for line in f]

    # Saving each column in a list
    mem_list = []
    # List to append mem_list to store it in one pandas dataframe
    file_list = []

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
            file_list.append(mem_list)
            mem_list = []

    exp_list.append(file_list)
    #

    # Making one big array for pandas dataframe
complete_data_array = np.concatenate(exp_list)
df = pd.DataFrame(complete_data_array,columns=column_headers)

# Saving results
if bool_save:
    saving_files(df,sample_data_folder_path,save_format,output_filename)

# Adding labels for each measurement
df = adding_measurement_label(df,n_datapoints)

# Analyze data: min, max, mean
# Analysing data
# Select only feasible points under sensor
list_selected_points = [2,3,4]
select = df.columns.get_level_values(1).isin(list_selected_points)
# We want to keep the column 'label'
select[-1] = True
df_analyse = df.loc[:, select]


df_analyse = pd.DataFrame(columns=column_headers,index=['Min','Max','Mean','std'])
df_analyse.drop(columns='Profiles_HostTime_start')

for label in pd.unique(df_analyse['label']):
    means = df[df.label == label].mean(numeric_only=True)
x_mean = means.Profiles_Velocity_X.to_list()
y_mean = means.Profiles_Velocity_Y.to_list()
z1_mean = means.Profiles_Velocity_Z1.to_list()
z2_mean = means.Profiles_Velocity_Z2.to_list()
# Calculate mean per measurement point

## General Plots
plot_histogram_delta_t(df)




print(datetime.now() - startTime)
