import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from datetime import datetime


def logs_preprocess(logs):
    # returns a list of 2 lists containing beginning and ending timestamps of the app use according to the log file
    start_timestamps = []
    end_timestamps = []
    label = ""
    with open(logs, 'r') as f:
        profilesList = [ re.split(r' ', line, maxsplit=4) for line in f.readlines() ]
        for line in profilesList : 
            if len(line) >= 2 and line[2] == 'U' :
                description = line[4].split(" ")[0]
                if description == "Start" : 
                    start_timestamps.append(datetime.strptime('23-'+line[0]+' '+line[1], '%y-%m-%d %H:%M:%S.%f'))
                    label = line[4].split(" ")[1]
                else : 
                    end_timestamps.append(datetime.strptime('23-'+line[0]+' '+line[1], '%y-%m-%d %H:%M:%S.%f'))           
    return [start_timestamps, end_timestamps, label]
            

def first_preprocess(data) :
    data.sort_values(by="Time", ascending=True,inplace =True)
    data.reset_index(drop=True, inplace = True)
    data["Time"] = pd.to_datetime(data.Time, unit='ms')
    reserved_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 65534, 65535]
    data=data[~data.rnti.isin(reserved_values)] # removes the reserved RNTIs
    return data

def second_preprocess(data,interval) : 
    # interval is a list containing beginning and ending timestamps
    # returns data where Time value is within the interval
    if ( len(interval[0]) == len(interval[1]) ) : 
        output = pd.DataFrame()
        for i in range(0,len(interval[0])) : 
            data_app = data[~data["Time"].lt(interval[0][i])] # removes the dates prior to the opening of the application
            data_app = data_app[~data_app["Time"].gt(interval[1][i])] # removes the dates after the opening of the application
            output = pd.concat([output,data_app])

    return output
    


if __name__ == '__main__' :
    if len(sys.argv) == 1 : 
        # no need to specify the application label if there is a log file
        print("Missing arguments : process_4G_file.py [Trace_file] [Log_file | datetime_begin(fmt YYYYMMDD-HHMMSS) datetime_end label_application]")
    
    else : 
        
        intervalles = []
        label = ""
        data = pd.read_csv(sys.argv[1])
        m = re.search('Trace(.+)_with', sys.argv[1])
        id_acquisition  = m.group(1)
        data = first_preprocess(data)

        if len(sys.argv) > 3 : 
            # case without log file, the beginning and ending dates are specified manually
            datetime_begin = datetime.strptime(sys.argv[2], '%Y%m%d-%H%M%S')
            datetime_end = datetime.strptime(sys.argv[3], '%Y%m%d-%H%M%S')
            intervalles = [[datetime_begin], [datetime_end]]
            label = sys.argv[4]

        elif len(sys.argv) == 3 : 
            # case with log file from the automation application (Automate)
            logs_processed = logs_preprocess(sys.argv[2])
            intervalles = logs_processed[0:2]
            label = logs_processed[2]
            data = first_preprocess(data)

        clean_data = second_preprocess(data,intervalles)
        clean_data["label"] = label  
        noise = pd.DataFrame()
        noise = data[~data.Time.isin(clean_data.Time)]
        noise["label"] = "noise"  

        full_data = pd.concat([clean_data,noise])
        full_data["id_acquisition"] = id_acquisition
        full_data.sort_values(by="Time", ascending=True,inplace =True)
        filename = './'+ label + id_acquisition + '.csv'
        full_data.to_csv(filename, encoding='utf-8', index=False)
        print(filename+" has been created.")

