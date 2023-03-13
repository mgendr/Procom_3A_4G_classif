import os
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

from tqdm import tqdm
from random import randint


from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler




DEFAULT_LENGTH_Value = 60


def format_data(filename, global_data):
    # used to adapt the format 
    # return a dictionnary : in the keys we have a global unique ID
    # in the values, we have the TBS values and the label
    data = pd.read_csv(filename)
    data["TBS_2"] = data["TBS_2"].replace(to_replace = -2, value = 0)
    data["label"] = data["label"].replace(to_replace = "pure_noise", value = "noise")
    data.set_index(pd.to_datetime(data.Time), inplace=True)
    
    id_acquis = data.id_acquisition.iloc[0] # unique for each file
    
    for val in tqdm(data.connection_id.unique()): # for each unique RNTI 
        current_data = data[data.connection_id==val] # We only keep the corresponding values
        label = current_data.label.value_counts(sort = True, ascending =False).index[0]# The label is the most frequent label
        global_data[f"{id_acquis}_{val}"]=(current_data.TBS_1, current_data.TBS_2,label) # We save TBS1, TBS2 and the label
        # With the key id_file + id_RNTI
    
    return global_data

def load_merge_datasets(path_directory) :
    # used to merge different dataset in in a single directory
    datasets = [i for i in os.listdir(path_directory) if i.endswith(".csv") ]
    global_data = {}
    for i, data in enumerate(datasets) :
        file = os.path.join(path_directory,data)
        print(f"Extracting {file}.... {i+1}/{len(datasets)}")
        global_data = format_data(file, global_data)
         
    return global_data

# ______Format in windows : ______

def reformat_data(global_data, length_value = DEFAULT_LENGTH_Value, step = 1) :
    #  split the data in windows of 60s
    # return a dictionnary : in the keys we have a global unique ID
    # in the values, we have the TBS values and the label
    reformated_data = {}

    for key, value in tqdm(global_data.items()) :
        TBS_1, TBS_2, label = value
       
        current_data = pd.concat([TBS_1, TBS_2],axis=1)
 
        current_data = current_data.resample('s').mean().interpolate()
        if len(current_data)>=length_value :
            
            count_seq_to_gen = int(np.floor((len(current_data)-length_value)/step))+1
            try :
            
                for i in range(count_seq_to_gen) :
                    current_window = current_data.iloc[i*step:i*step+length_value]      
                    reformated_data[f"{key}_{i*step}"]=(current_window.TBS_1, current_window.TBS_2,label)
            except :
                print(f"count_seq_to_gen {count_seq_to_gen}")
                print(f"len(current_data) {len(current_data)}")
                print(f"length_value {length_value}")
                print(f"step {step}")
                print(a)
                    
    return reformated_data

def adapt_to_dataframe(data) :
    # transforms the dic in df

    futur_df = dict([(key,list(value[0].values)+list(value[1].values) + [value[2]]) for key, value in data.items()])
    futur_df = pd.DataFrame(futur_df).T
    
    futur_df.columns = list(futur_df.columns[:-1])+["label"]
    
    return futur_df


def load_and_preprocess_agg_window(directory_data_test, length_value = DEFAULT_LENGTH_Value, step = 1 ) :
    
    dico = reformat_data(load_merge_datasets(directory_data_test),length_value = length_value, step = step )
    return adapt_to_dataframe(dico)

# ______Format in metrics : ______


def compute_metrics(data) :
    list_rows = []
    for key,value in tqdm(data.items()) : # Pour chaque time serie
        
        new_row = first_layer(value[0],value[1])
        
        new_row["label"] = value[2]
        new_row["id"] = key
        
        list_rows.append(new_row)
    data_final = pd.DataFrame.from_records(list_rows)
    data_final.set_index("id",drop = True, inplace=True)
    # concatenate new rows  
    # return concatenation
    return data_final

def first_layer(TBS_1,TBS_2):
    metrics = {}
    cum_TBS_1 = TBS_1.cumsum()
    cum_TBS_2 = TBS_2.cumsum()
    # apply scaler
    
    # Here we apply models on cumulated sum
    models = ["reg_lin" ]#, "reg_iso"] #, "reg_pol"]
    
    for model in models : 
        metrics = apply_simple_model(model, cum_TBS_1, metrics,added_name ="TBS_1")
        metrics = apply_simple_model(model, cum_TBS_2, metrics,added_name ="TBS_2")
        
    # Here we can measure other metrics on none cumulated sum
    
    # Q1 to obtain plateau. median, mean Q3 , other percentiles(10, 90)... same usage I hope.
    # intuition = un plateau = TBS à 0 pendant un certain temps, ainsi x% du temps le TBS vaut 0 donc faible percentile à 0 
    percentiles = [0.1,0.25,0.5,0.75,0.9]
    to_get = {"min" : "p0", "10%" : "p10", "25%" : "p25", "50%" : "p50", "75%" : "p75",
              "90%" : "p90", "max" : "p100", "mean" : "mean", "std" : "std"}
    
    stats_TBS_1 = TBS_1.describe(percentiles=percentiles)
    for desc_name, new_name in to_get.items() :
        metrics[f"{new_name}_TBS_1"] =  stats_TBS_1.loc[desc_name]
        
    stats_TBS_2 = TBS_2.describe(percentiles=percentiles)
    for desc_name, new_name in to_get.items() :
        metrics[f"{new_name}_TBS_2"] =  stats_TBS_2.loc[desc_name]
        
        
    # add duration
    
    return metrics

def apply_simple_model(model, data, metrics,added_name ="") :
    current_data = data.dropna()

    y = current_data.values
    X = pd.to_datetime(data.index).astype(int)/ 10**9
    first_time = X[0]
    X-=first_time  
    X = np.array(X).reshape(-1, 1)
    
    if model =="reg_lin" :
        reg = LinearRegression()
        reg.fit(X, y)
        metrics[f"reg_lin_R2_{added_name}"] = reg.score(X, y)
        metrics[f"reg_lin_coef_A_{added_name}"] = reg.coef_[0]
        
    elif model == "reg_iso" :
        iso_reg = IsotonicRegression(increasing = True)
        iso_reg.fit(X, y)
        metrics[f"reg_iso_R2_{added_name}"] = iso_reg.score(X, y) # non fonctionnel
    
    return metrics



def load_and_preprocess_agg_metrics(directory_data_test) :

    data = compute_metrics(load_merge_datasets(directory_data_test))
    
    return data


class Scaler_Metrics:
    scaler = None
    
    def fit_transform(self, metrics_data) :
        x_unscaled = metrics_data[list(metrics_data.columns[:-1])].values
        y = metrics_data["label"]
        
        current_scaler = StandardScaler()
        x = current_scaler.fit_transform(x_unscaled)
        self.scaler = current_scaler
        new_df = pd.DataFrame(x,columns = list(metrics_data.columns[:-1]))
        
        
        new_df["label"] = y.values
        new_df["id"] = metrics_data.index
        new_df.set_index("id", drop = True, inplace = True)

        return new_df
    def transform(self, metrics_data) :
        x_unscaled = metrics_data[list(metrics_data.columns[:-1])].values
        y = metrics_data["label"]
        

        x = self.scaler.transform(x_unscaled)

        new_df = pd.DataFrame(x,columns = list(metrics_data.columns[:-1]))
        print(y)
        new_df["label"] = y.values
        new_df["id"] = metrics_data.index
        new_df.set_index("id", drop = True, inplace = True)
        
        return new_df
        
    






