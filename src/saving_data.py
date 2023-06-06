def saving_files(df,sample_data_folder_path,save_format,output_filename):
    
    output_destination = sample_data_folder_path + save_format + '/' + output_filename
    if save_format == 'csv':
        output_destination = output_destination + '.csv'
        df.to_csv(output_destination)
    elif save_format == 'h5':
        output_destination = output_destination + '.h5'
        df.to_hdf(output_destination,key='df')
    else:
        None