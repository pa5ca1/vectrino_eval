import numpy as np
import pandas as pd
import os

# For timing
from datetime import datetime
startTime = datetime.now()

# get current working directory
cwd = os.getcwd()

# Load file with column headers
file_headers = cwd + '/column_header_2.txt'

# Load files


folder_path = cwd + '/../sample_data/'

files = os.listdir(folder_path)
file_list = [file for file in files if file.endswith('.dat')]

for file_sample_name in file_list:
    file_sample = cwd + '/../sample_data/' + file_sample_name
    # We delete the .dat at the end and put everything else together
    output_filename = os.path.splitext(file_sample_name)[0]
    output_destination = folder_path + output_filename + '.h5'

    # Load headers
    with open(file_headers) as f:
        headers = [line for line in f]
    # Remove '\n'
    headers = [head.replace('\n','') for head in headers]

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

        # Do not use the values with 'VelocityHeader' in name
        if 'VelocityHeader' not in column_name and 'BottomCheck_' not in column_name:
        
            if '[' not in parts[1]:
                # unix time
                after_colon = float(parts[1].strip().replace(",", "."))
            else:
                # experimental values
                after_colon = [x.replace(',', '.') for x in parts[1].split()][1:-1]
                # Check if we read a status which is only an integer values
                if len(after_colon) > 1:
                    # Not a status
                    after_colon = np.array([float(x) for x in after_colon])
                else:
                    # Status
                    after_colon = after_colon[0]
            mem_list.append(after_colon)
            if column_name == 'Profiles_AveragedPingPairs':
                big_list.append(mem_list)
                mem_list = []


    df = pd.DataFrame(big_list,columns=headers)
    #df.to_csv(output_destination)

    print('Saving: ', file_sample_name)
    df.to_hdf(output_destination,key='df')
    #print(df.head())
    #df.to_csv(output_filename)

df2 = pd.read_hdf(folder_path + '/VectrinoData.144.23.Vectrino Profiler.00006.h5')

print(datetime.now() - startTime)