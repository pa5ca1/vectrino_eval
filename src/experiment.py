import os
import numpy as np
import pandas as pd

class Experiment:

    def __init__(self,name,pos_data_name):
        

        self.cwd =  os.getcwd()
        self.name = name
        self.path_result_data = self.cwd + '/results_data/'
        self.path_raw_data = self.cwd + '/sample_data/' + self.name + '/'
        self.path_position_data = self.cwd + '/position_times/'
        self.position_data_name = pos_data_name

        self.list_of_files = self._get_all_files()

        self.position_traverse, self.time_traverse = self._get_positions()
        self.grid_dimension = self._get_grid_dim()
        self.n_measurements = len(self.position_traverse)-1

        
        self.lines_to_extract = ['Profiles_Velocity_X','Profiles_Velocity_Y',
                        'Profiles_Velocity_Z1','Profiles_Velocity_Z2',
                        'Profiles_HostTime_start']
        
        self.n_sample_depth = self._get_sample_depth()
        
        self.column_dict = {'Profiles_HostTime_start':1,
                'u_x': self.n_sample_depth,
                'u_y': self.n_sample_depth,
                'u_z1' :self.n_sample_depth,
                'u_z2' :self.n_sample_depth}
        
        self.raw_data = self._extract_raw_data()
        self.trimmed_data = self._trim_data()
        self.statistics_values = self._calculate_mean_per_measurement()
        self.turbulunce_intensity = self._calculate_turbulence_intensity()


    def _get_grid_dim(self):
        if self.position_data_name == 'pos0':
            return (11,7)
        if self.position_data_name == 'pos1':
            return (9,7)

    def _calculate_turbulence_intensity(self):

        I = []
        labels = self.trimmed_data.label.unique()
        for label in labels:
            u_x_mean = self.statistics_values[self.statistics_values.label == 
                                              label].u_x_bar.iloc[0]
            u_y_mean = self.statistics_values[self.statistics_values.label == 
                                              label].u_y_bar.iloc[0]
            u_z1_mean = self.statistics_values[self.statistics_values.label == 
                                               label].u_z1_bar.iloc[0]
            u_z2_mean = self.statistics_values[self.statistics_values.label == 
                                               label].u_z2_bar.iloc[0]

            u_x = self.trimmed_data[self.trimmed_data.label == 
                                         label].u_x.to_numpy().reshape(-1)
            u_dash_sq_mean_x = np.sqrt(np.mean((u_x-u_x_mean)**2))

            u_y = self.trimmed_data[self.trimmed_data.label == 
                                         label].u_y.to_numpy().reshape(-1)
            u_dash_sq_mean_y = np.sqrt(np.mean((u_y-u_y_mean)**2))

            u_z1 = self.trimmed_data[self.trimmed_data.label == 
                                          label].u_z1.to_numpy().reshape(-1)
            u_dash_sq_mean_z1 = np.sqrt(np.mean((u_z1-u_z1_mean)**2))

            u_z2 = self.trimmed_data[self.trimmed_data.label == 
                                          label].u_z2.to_numpy().reshape(-1)
            u_dash_sq_mean_z2 = np.mean((u_z2-u_z2_mean)**2)   

            u_dash = np.sqrt((u_dash_sq_mean_x + u_dash_sq_mean_y + u_dash_sq_mean_z1)/3)
            u_hat = np.sqrt(u_x_mean**2 + u_y_mean**2 + u_z1_mean**2)

            I.append(u_dash/u_hat)
        I = np.array(I).reshape(self.grid_dimension)
        I = self._reshape_turbulence(I)
        return I

    def _reshape_turbulence(self,I):
        
        I[1::2, :] = I[1::2, ::-1]
        I = np.flipud(I)
        return I


    def _calculate_mean_per_measurement(self):

        u_x = []
        u_y = []
        u_z1 = []
        u_z2 = []
        std_vel = []
        t_intervall = []
        labels = self.trimmed_data.label.unique()


        for label in labels:
            mean_values = self.trimmed_data[self.trimmed_data.label == label].mean(numeric_only=True)
            std_values = self.trimmed_data[self.trimmed_data.label == label].std(numeric_only=True)
            u_x.append(mean_values.u_x.mean())
            u_y.append(mean_values.u_y.mean())
            u_z1.append(mean_values.u_z1.mean())
            u_z2.append(mean_values.u_z2.mean())

            t1 = self.trimmed_data[self.trimmed_data.label == label].Profiles_HostTime_start.iloc[-1]
            t2 = self.trimmed_data[self.trimmed_data.label == label].Profiles_HostTime_start.iloc[0]
            t_intervall.append((t2,t1))
            x_std = std_values.u_x.std()
            y_std = std_values.u_y.std()
            z1_std = std_values.u_z1.std()
            z2_std = std_values.u_z2.std()
            std_vel.append([x_std,y_std,z1_std,z2_std])


        mean_df = pd.DataFrame()
        mean_df['u_x_bar'] = u_x
        mean_df['u_y_bar'] = u_y
        mean_df['u_z1_bar'] = u_z1
        mean_df['u_z2_bar'] = u_z2
        std_vel = np.array(std_vel)
        mean_df['x_std'] = std_vel[:,0]
        mean_df['y_std'] = std_vel[:,1]
        mean_df['z1_std'] = std_vel[:,2]
        mean_df['z2_std'] = std_vel[:,3]
        mean_df['time_intervall'] = t_intervall
        mean_df['label'] = labels

        return mean_df

    def _trim_data(self,list_selected_points = [2,3],delta_t = [10,20]):

        select = self.raw_data.columns.get_level_values(1).isin(list_selected_points)
        # We want to keep the column 'label'
        select[-1] = True
        # We want to keep timestamps
        #select[0] = True
        df = self.raw_data.loc[:, select].copy()
        # We want to keep timestamps
        df['Profiles_HostTime_start'] = self.raw_data.Profiles_HostTime_start.to_numpy()

        # cut in time
        labels = np.flip(df.label.unique())

        for label in labels:
            t_0 = df[df.label == label].Profiles_HostTime_start.to_numpy()[0]
            t_3 =df[df.label == label].Profiles_HostTime_start.to_numpy()[-1]
            t_1 = t_0 + delta_t[0]
            t_2 = t_3 - delta_t[1]
            i_0 = df[(df.Profiles_HostTime_start>=t_0) & (df.Profiles_HostTime_start<t_1)].index
            i_1 = df[(df.Profiles_HostTime_start>t_2) & (df.Profiles_HostTime_start<=t_3)].index
            df = df.drop(i_1).drop(i_0)
        
        return df



    def save_data(self,key_word):
       
        if key_word == 'turbulence_intensity':
            place_to_save = self.path_result_data + self.name + '_turb_int'
            data = self.turbulunce_intensity
        
        try:
            np.save(place_to_save,data)
        except:
            if not isinstance(data, np.ndarray):
                raise Exception("Not an numpy array") 
            else:
                raise Exception("Unknown Exception")



    def _get_all_files(self):

        files = os.listdir(self.path_raw_data)
        list_of_all_files = [file for file in files if file.endswith('.dat')]

        # Sorting the files alphabetically
        list_of_all_files.sort()
        list_of_all_files = list_of_all_files[:-1]

        return list_of_all_files
    
    def _get_positions(self):

        file_to_open = self.path_position_data + self.position_data_name + '.txt'
        
        X = []
        T = []

        with open(file_to_open, 'r') as file:
            for line in file:
                values = line.strip().replace(',', '.').split()
                if len(values) == 4:
                    try:
                        val1, val2, val3, val4 = map(str, values)
                        X.append([val1, val2, val3])
                        T.append(val4)
                    except ValueError:
                        print(f"Invalid values in line: {line}")

        return np.array(X), np.array(T)

    def _get_sample_depth(self):

        file_to_open = self.path_raw_data + self.list_of_files[0]
        with open(file_to_open) as f:
            lines = [line for line in f]

        for line in lines:
            # Each line consits of two parts: First a name and
            # second the value, seperated by colon
            parts = line.split(":")
            column_name = parts[0].strip()
            column_name = column_name.split()[0]
            if '['  in parts[1]:
                after_colon = [x.replace(',', '.') for 
                                       x in parts[1].split()][1:-1]
            
                return len(after_colon)



    def _extract_raw_data(self):

        # How to name the columns in pandas dataframe
        main_column = [key for key, value in self.column_dict.items() for 
                       _ in range(value)]
        sub_column = []
        for key, value in self.column_dict.items():
            sub_column.extend(list(range(1, value + 1)))
        column_headers = [np.array(main_column),np.array(sub_column)]

  
        print('#'*50)
        print('#'*50)
        print('Experiment: ' + self.name)
        print('No. of measurements: ' + str(self.n_measurements))
        print('Start to load raw data ...')
        print('#'*50)
        print('#'*50)

        exp_list = []
        for i,file_sample_name in enumerate(self.list_of_files):
            print('Sampling: ' + str(i+1) + 
                  ' of ' + str(self.n_measurements) + 
                  ' | Name: ' + file_sample_name)
            # Choosing the correct file
            file_sample = self.path_raw_data + file_sample_name
            
            # We delete the string '.dat' at the end and put everything else together
            # TODO May delete later
            #output_filename = os.path.splitext(file_sample_name)[0]

            # extract lines from sample file
            with open(file_sample) as f:
                lines = [line for line in f]

            # Saving each column in a list
            mem_list = []
            # List to append mem_list to store it in one pandas dataframe
            file_list = []

            for line in lines:
            # Each line consits of two parts: First a name and
            # second the value, seperated by colon
                parts = line.split(":")
                column_name = parts[0].strip()
                column_name = column_name.split()[0]


                # Do not use the values with 'VelocityHeader' in name
                if column_name in self.lines_to_extract:
                
                    if '[' not in parts[1]:
                        # unix time
                        after_colon = float(parts[1].strip().replace(",", "."))
                        mem_list.append(after_colon)
                    else:
                        # experimental values
                        after_colon = [x.replace(',', '.') for 
                                       x in parts[1].split()][1:-1]
                        # Check if we read a status which is 
                        # only an integer values or not
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
        df = self._adding_measurement_label(df, self.n_measurements)

        return df

    def _adding_measurement_label(self,df,n_datapoints):
        '''Adds a label for each meassurement if a) a sample is taken or 
        b) the traverse is moving.
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
        list_periods = np.array([start_time+i*t_per_datapoint for 
                                 i in range(n_datapoints+1)])
        # We use .squezze to change from pandas dataframe to timeseries
        # because 'get_label' needs a timeseries
        df['label'] = df['Profiles_HostTime_start'].squeeze().apply(self._get_label,l=list_periods)

        return df
    
    def _get_label(self,timestamp,**kwargs):
        ''' Labeling each '''

        label_list = kwargs.get('l', None)

        for i in range(len(label_list)-1):
            if label_list[i] <= timestamp <= label_list[i+1]:
                # Convert to corresponding label ('A', 'B', 'C', ...)
                return chr(65 + i)  
        return 'Test'