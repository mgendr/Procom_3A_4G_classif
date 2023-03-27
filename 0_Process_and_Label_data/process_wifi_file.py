import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from datetime import datetime

# dc:0b:34:bd:68:55 or LgElectr_bd:68:55 = WiFi MAC address (LG NEXUS) 
# c0:bd:d1:e7:97:f4 or SamsungE_e7:97:f4 = WiFi MAC address (SAMSUNG GALAXY S5) 
mac_addresses = ["dc:0b:34:bd:68:55", "c0:bd:d1:e7:97:f4"]
id_acquisitionHMS = ""

def conn_id(mac_address) : 
    # returns an "arbitrary" connection_id for the specified MAC address 
    # this connection_id is used in order to fit the same format as 4G decoded captures
    for i in range (0, len(mac_addresses)):
        if mac_address == mac_addresses[i] : return str(14+i)

def first_preprocess(data) :
    global id_acquisitionHMS
    data.sort_values(by="Time", ascending=True,inplace =True)
    data.reset_index(drop=True, inplace = True)
    data["Time"] = pd.to_datetime(data.Time, format="%Y-%m-%d %H:%M:%S,%f")
    id_acquisitionHMS = data["Time"][0].strftime("%H%M%S")
    data = data[data[["wlanSource","wlanDest"]].isin(mac_addresses).any(axis=1)] # keeps the packets that only involve our 2 smartphones 
    data["downlink"] = 0 
    data["uplink"] = 0 
    data.loc[(data.wlanSource.isin(mac_addresses)) & (~data.wlanDest.isin(mac_addresses)), "uplink"]  = (data["Length"])*8
    data.loc[(data.wlanDest.isin(mac_addresses)) & (~data.wlanSource.isin(mac_addresses)), "downlink"]  = (data["Length"])*8
    data.loc[(data.wlanSource.isin(mac_addresses)) & (~data.wlanDest.isin(mac_addresses)), "connection_id"]  = data.wlanSource
    data.loc[(data.wlanDest.isin(mac_addresses)) & (~data.wlanSource.isin(mac_addresses)), "connection_id"]  = data.wlanDest
    data = data.loc[(~data.wlanSource.isin(mac_addresses)) | (~data.wlanDest.isin(mac_addresses))]
    data["connection_id"] = data["connection_id"].apply(conn_id)

    return data

def second_preprocess(data,interval) : 
    # interval is a list containing beginning and ending timestamps
    # returns data where Time value is within the interval
    if ( len(interval) == 2 and len(interval[0]) == len(interval[1]) ) : 
        output = pd.DataFrame()
        for i in range(0,len(interval[0])) : 
            data_app = data[~data["Time"].lt(interval[0][i])] # removes the dates prior to the opening of the application
            data_app = data_app[~data_app["Time"].gt(interval[1][i])] # removes the dates after the opening of the application
            output = pd.concat([output,data_app])

    return output

if __name__ == '__main__' :
    print(len(sys.argv))
    if len(sys.argv) != 3 and len(sys.argv) != 5 : 
        print("Bad arguments (case with noise values) : process_wifi_file.py [Trace_file (fmt WifiYYYYMMDD_*.csv)] [datetime_begin_app(fmt YYYYMMDD-HHMMSS) datetime_end_app] [label_application]")
        print("Bad arguments (case without noise) : process_wifi_file.py [Trace_file (fmt WifiYYYYMMDD_*.csv)] [label_application]")
    
    else : 
        
        interval = []
        label = ""
        data = pd.read_csv(sys.argv[1])
        m = re.search('Wifi(.+)_', sys.argv[1])
        id_acquisition  = m.group(1)
        data = first_preprocess(data)

        if len(sys.argv) == 5 : 
            datetime_begin = datetime.strptime(sys.argv[2], '%Y%m%d-%H%M%S')
            datetime_end = datetime.strptime(sys.argv[3], '%Y%m%d-%H%M%S')
            interval = [[datetime_begin], [datetime_end]]
            label = sys.argv[4]
            data_app = second_preprocess(data,interval)
            data_app["label"] = label 
            data_noise = pd.DataFrame()
            data_noise = data[~data.Time.isin(data_app.Time)]
            data_noise["label"] = "noise"
            full_data = pd.concat([data_app,data_noise])
          
        elif len(sys.argv) == 3 :
            label = sys.argv[2]
            full_data = data
            full_data["label"] = label
        

        full_data.sort_values(by="Time", ascending=True,inplace =True)
        full_data["id_acquisition"] = id_acquisition +'_' + id_acquisitionHMS
        filename = './wifi-'+ label + id_acquisition +'_' + id_acquisitionHMS + '.csv'
        full_data.to_csv(filename, encoding='utf-8', index=False)
        print(filename+" has been created.")
