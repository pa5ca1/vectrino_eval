import numpy as np
import pandas as pd
import os

from analyse_data import adding_measurement_label
from analyse_data import select_points_in_space
from analyse_data import select_points_in_time

def load_exp(cwd,exp_name, file_name = None):
    ## Load files
    load_all = True
    sample_data_folder_path = cwd + '/../sample_data/' + exp_name + '/'
    # only use if one specific file should be used
    if load_all:
        files = os.listdir(sample_data_folder_path)
        list_of_all_files = [file for file in files if file.endswith('.dat')]
    else:
        list_of_all_files = [file_name]
    
    # Sorting the files alphabetically
    list_of_all_files.sort()
    list_of_all_files = list_of_all_files[:-1]
    # Number of all datapoints
    n_datapoints = len(list_of_all_files)




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
    print('#'*50)
    print('#'*50)
    print('Experiment: ' + exp_name)
    print('#'*50)
    for i,file_sample_name in enumerate(list_of_all_files):
        print('Sampling: ' + str(i+1) +' of ' + str(n_datapoints) + ' | Name: ' + file_sample_name)
        # Choosing the correct file
        file_sample = cwd + '/../sample_data/' + exp_name + '/' + file_sample_name
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

    print('#'*50)
    print('End of experiment')
    print('#'*50)
    # Making one big array for pandas dataframe
    complete_data_array = np.concatenate(exp_list)
    df = pd.DataFrame(complete_data_array,columns=column_headers)

    # Adding labels for each measurement
    df = adding_measurement_label(df,n_datapoints)

    return df


def cut_data(df,list_selected_points = [2,3],  delta_t = [10,20]):
    
    # Select points in space
    df_select_space = select_points_in_space(df,list_selected_points)
    # Select relative time point
    # cut 10 second in the front and 20 seconds before end of measurement
    df_select_space_and_time = select_points_in_time(df_select_space,delta_t)

    return df_select_space_and_time