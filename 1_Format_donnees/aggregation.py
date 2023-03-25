import os
import pandas as pd
import time
import numpy as np

from tqdm import tqdm

from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler




DEFAULT_LENGTH_Value = 60 # the default window length (in seconds)


def format_data(filename, global_data):
    """
    used to adapt the format a raw csv file and to merge it with the previous ones

    Parameters
    ----------
    filename : str
        path to the csv file
    global_data : dic 
        Same format as the returned dic. Contains the previously loaded files.

    Returns
    -------
    global_data : dic with elements as follow `id : (TBS,label)` 
        id : unique id for a specific track (unique accross all files)
        TBS : Dataframe with 2 columns : TBS_down, TBS_up. Timestamps as index
        label : the corresponding label for the track

    """
    
    # read
    data = pd.read_csv(filename)
    
    
    # We have to replace some values in the TBS columns (instead of 0, the column contains -2)
    data["TBS_2"] = data["TBS_2"].replace(to_replace = -2, value = 0)
    data["TBS_sum"] = data["TBS_2"] + data["TBS_1"] #☻ the TBS values is the sum of TBS 1 & TBS 2

    
    # Now let's find if it's TBS_up or TBS_down ?
    uplink = data.format == 0. # a mask of booleans which is equal to 1 if the current line corresponds to an upload
    
    data["TBS_up"] = uplink*data["TBS_sum"]# We apply the mask in to get the TBS_up
    data["TBS_down"] = (1-uplink)*data["TBS_sum"] # Same for TBS_down

    #data.drop(colmns = ["TBS_1", "TBS_2", "TBS_sum"], inplace = True)
    data["label"] = data["label"].replace(to_replace = "pure_noise", value = "noise")# correct a duplicate label name
    

    data.set_index(pd.to_datetime(data.Time), inplace=True)# put the time as the index
    

    id_acquis = data.id_acquisition.iloc[0] # unique for each file
    

    for val in tqdm(data.connection_id.unique()): # for each unique RNTI 
        current_data = data[data.connection_id==val] # We only keep the corresponding values
        label = current_data.label.value_counts(sort = True, ascending =False).index[0]# The label is the most frequent label
        global_data[f"{id_acquis}_{val}"]=(current_data[["TBS_down", "TBS_up"]],label) # We save TBS up and down & the label
        # With the key id_file + id_RNTI
        
    return global_data

def load_merge_datasets(path_directory) :
    """
    For a given folder, load all the csv files contained in it and applies some preprocesses, as track splitting.
    
    

    Parameters
    ----------
    path_directory : str
        path to the folder which contains the csv files

    Returns
    -------
    global_data : dic with elements as follow `id : (TBS,label)` 
        id : unique id for a specific track (unique accross all files) id_file + id_RNTI
        TBS : Dataframe with 2 columns : TBS_down, TBS_up. Timestamps as index
        label : the corresponding label for the track

    """
    # used to merge different dataset in in a single directory
    datasets = [i for i in os.listdir(path_directory) if i.endswith(".csv") ] # all the files names
    global_data = {} # Will store the data
    for i, data in enumerate(datasets) :
        file = os.path.join(path_directory,data) # the name merged with the path
        print(f"Extracting {file}.... {i+1}/{len(datasets)}") # Some informations while loading data
        global_data = format_data(file, global_data) # merge the current file with the previously loaded data
         
    return global_data

# _____________________________________Format in windows : ______________________________________________

def reformat_resample_data(global_data, length_value = DEFAULT_LENGTH_Value, step = 1) :
    """
    splits the data in windows of a specific length (in seconds).
    we will also resample the time series with a frequency of 1 measure/s. 
    Used to have smoother time serie
    

    Parameters
    ----------
    global_data : dic in the format returned in load_merge_datasets
        all the loaded data
    length_value : int, optional
        Size in seconds of the windows. The default is DEFAULT_LENGTH_Value.
    step : int, optional
        step decay between each windows. The default is 1.

    Returns
    -------
    reformated_data : dic
        same format of results as load_merge_datasets
        the id is slighty different : id_file + id_RNTI + id_window (instead of id_file + id_RNTI)
            

    """
    
    reformated_data = {} # Wil store the data

    for key, value in tqdm(global_data.items()) : # for each item in the dictionnary : 

        TBS, label = value # get TBS & label
 
        current_data = TBS.resample('s').sum().interpolate() # resample for each second and fill each value with Nan(no measure in a specific interval)
        if len(current_data)>=length_value : # if we have enough data to create at least one window 
            
            # let's count the number of windows that we can create
            count_seq_to_gen = int(np.floor((len(current_data)-length_value)/step))+1

            for i in range(count_seq_to_gen) :
                current_window = current_data.iloc[i*step:i*step+length_value]  # extract the current window    
                reformated_data[f"{key}_{i*step}"]=(current_window,label) # save it

                    
    return reformated_data

def adapt_to_dataframe(data) :
    """
    Transforms the data with the dictionnary format(cf load_merge_datasets) to a DataFrame

    Parameters
    ----------
    data : dic (cf load_merge_datasets) id : (TBS,label)
        Same format as the dictionnary. We must have TBS with a fixed size.

    Returns
    -------
    futur_df : DataFrame Pandas
        the Dataframe which will be used to train/test models.

    """
    
    
    # concat TBS up and down in a single line
    futur_df = dict([(key,list(value[0].TBS_up.values)+list(value[0].TBS_down.values) + [value[1]]) for key, value in data.items()])
    # Transform in Dataframe and rotate it
    futur_df = pd.DataFrame(futur_df).T
    
    # rename the label column
    futur_df.columns = list(futur_df.columns[:-1])+["label"]
    
    return futur_df


def load_and_preprocess_agg_window(directory_data_test, length_value = DEFAULT_LENGTH_Value, step = 1 ) :
    """
    run each step to compute the resampled data format

    Parameters
    ----------
    directory_data_test : str
        path to the folder which contains the csv files
    length_value : int, optional
        Size in seconds of the windows. The default is DEFAULT_LENGTH_Value.
    step : int, optional
        step decay between each windows. The default is 1.

    Returns
    -------
    DataFrame Pandas
        the Dataframe which will be used to train/test models.

    """
    print( "Load datasets...")
    datasets = load_merge_datasets(directory_data_test)
    
    print("Reformat...")
    dico = reformat_resample_data(datasets,length_value = length_value, step = step )
    return adapt_to_dataframe(dico)

# ______Format in metrics : ______


def compute_metrics(data, min_size = 15) :
    """
    Compute the metrics over the whole data

    Parameters
    ----------
    data : dic (cf load_merge_datasets)
        Data to process
    min_size : int, optional
        minimum size of a track (in count of record). The default is 15.

    Returns
    -------
    data_final : DataFrame Pandas
        the Dataframe which will be used to train/test models.

    """
    list_rows = [] # store the futur df
    for key,value in tqdm(data.items()) : # for each time serie
        
        TBS = value[0]
        
        if len(TBS)>min_size : # we check the condition on the minimum size 
            new_row = compute_metrics_for_a_track(TBS.TBS_up)# Compute metrics on TBS up
            new_row_down = compute_metrics_for_a_track(TBS.TBS_down)# Compute metrics on TBS down
            new_row.update(new_row_down) # merge the results
            
            new_row["label"] = value[1] # add the label
            new_row["id"] = key # add the unique id
        
            list_rows.append(new_row) # save the new row
            
    data_final = pd.DataFrame.from_records(list_rows) # transforms the set of rows in a dataframe
    data_final.fillna(0, inplace=True) # fill the nan values with 0. Useful for some metrics which can't be computed.
    data_final.set_index("id",drop = True, inplace=True) # set the id as the index

    return data_final

def compute_metrics_for_a_track(TBS):
    """
    compute the metrics for a single track

    Parameters
    ----------
    TBS : Serie Pandas
        Conatains the data to proceed

    Returns
    -------
    metrics : dic metric_name : value
        a record of each metrics for the current data

    """
    metrics = {}
    
    # Here we apply model on cumulated sum, the model is a linear regression
    metrics = apply_simple_model(TBS.cumsum(), metrics,added_name =TBS.name)

    TBS = TBS[TBS!=0] #remove the 0 values 

        
    # Q1 to obtain plateau. median, mean Q3 , other percentiles(10, 90)... same usage I hope.
    
    percentiles = [0.1,0.25,0.5,0.75,0.9]
    to_get = {"min" : "p0", "10%" : "p10", "25%" : "p25", "50%" : "p50", "75%" : "p75",
                "90%" : "p90", "max" : "p100", "mean" : "mean", "std" : "std"}
    
    if len(TBS)==0: # if we don't have any value on the up or down channel
        for desc_name, new_name in to_get.items() :
            metrics[f"{new_name}_{TBS.name}"] =  0
        
    else :  # else we can compute the basic metrics
        stats_TBS = TBS.describe(percentiles=percentiles) # provides metrics
        for desc_name, new_name in to_get.items() :
            metrics[f"{new_name}_{TBS.name}"] =  stats_TBS.loc[desc_name] 

    metrics[f"sum_{TBS.name}"] = TBS.sum() # the sum metric
    
    # the metric : trafic averaged by the duration
    if len(TBS)>1 : # we need at least 2 record, in order to have a duration > 0
        duration = (TBS.index[-1]-TBS.index[0]).total_seconds() # compute duration of the track
        metrics[f"mean_per_time_{TBS.name}"] = TBS.sum()/duration
    else :
        metrics[f"mean_per_time_{TBS.name}"] = 0

        
     
        
    # add duration
    
    return metrics

def apply_simple_model( data, metrics,added_name ="") :
    """
    Apply a model, here a Linear Regression

    Parameters
    ----------
    data : Serie Pandas
        The TBS values
    metrics : dic
        already recorded metrics
    added_name : str, optional
        suffixe to add at the end of the metric name. The default is "".

    Returns
    -------
    metrics : dic
        updated dictionnary of metrics

    """
    if len(data)>1 : #if we have enough data to compute the linear regression
        current_data = data.dropna() # drop nan values if any

        y = current_data.values # get the array values
        X = pd.to_datetime(data.index).astype(int)/ 10**9 # transforms the time stamps in an integer axis
        first_time = X[0]
        X-=first_time # set the origin of the time axis at 0
        X = np.array(X).reshape(-1, 1) #reshape in the proper format 

        reg = LinearRegression()
        reg.fit(X, y) # run the model

        reg_score, reg_coef = reg.score(X, y), reg.coef_[0] # get the metrics
    else : #else we set the metrics to 0
        reg_score, reg_coef = 0,0

    metrics[f"reg_lin_R2_{added_name}"] = reg_score
    metrics[f"reg_lin_coef_A_{added_name}"] = reg_coef

    return metrics

def split_in_windows_duration(data, window_size, step , min_duration = 30):
    """
    split the data in windows for the format in metrics

    Parameters
    ----------
    data : dic (cf load_merge_datasets)
        the data to process
    window_size : int
        the size of the window in seconds
    step : int
        step decay between each windows.
    min_duration : int, optional
        minimum duration of the track in seconds. The default is 30.

    Returns
    -------
    new_data : dic (cf load_merge_datasets)
        the new tracks obtained

    """
    new_data = {}
    
    for key, value in tqdm(data.items()): # for each time serie
        TBS, label = value[0], value[1] # get TBS values and label
        
        total_duration = (TBS.index[-1] - TBS.index[0]).seconds # compute the total duration of the track
        
      
        
        if total_duration>=min_duration : # if the constraint on minimum duration is respected
     
            count_seq_to_gen = int(np.floor((total_duration-window_size)/step))+1 # count number of futur generated tracks

            
            for i in range(count_seq_to_gen) : # for each one of thoses 
                beginning = TBS.index[0]+pd.Timedelta(step*i, "s") # beginning time
                end = TBS.index[0] + pd.Timedelta(step*i + window_size, "s") # ending time
                
                current_window = TBS.iloc[(TBS.index >=beginning) & (TBS.index<end)] # get the corresponding values
                new_data[f"{key}_{i*step}"]=(current_window,label) # save it with a new id
           
        

        
    return new_data


def load_and_preprocess_agg_metrics(directory_data_test, window_size = None, step = 1, min_duration = 30) :
    """
    Process all the steps in order to obtain the format in metrics
    Can compute both with & without windows

    Parameters
    ----------
    directory_data_test : str
        path to the folder which contains the csv files
    window_size : int, optional
        the size of the window in seconds. if set to None, don't compute windows. The default is None.
    step : int, optional
        step decay between each windows. Only used in windows mode(window_size!=None). The default is 1.
    min_duration : int, optional
        minimum duration of the track in seconds. Only used in windows mode(window_size!=None) The default is 30.

    Returns
    -------
    data : DataFrame Pandas
        the Dataframe which will be used to train/test models.

    """
        
    print( "Load datasets...")
    datasets = load_merge_datasets(directory_data_test)
    
    if not window_size is None :
        print("Splitting the data in windows...")
        time.sleep(.1)
        datasets = split_in_windows_duration(datasets, window_size, step, min_duration = min_duration )
        
    
    print("Computing metrics and statistics...")
    time.sleep(.1)  
    data = compute_metrics(datasets)
    return data




class Scaler_Metrics:
    """
    Scales Dataset
    """
    
    
    scaler = None
    
    def fit_transform(self, metrics_data) :
        """
        scale data(for training dataset) and store the scaling parameters

        Parameters
        ----------
        metrics_data : Pandas DataFrame
            training data to scale 

        Returns
        -------
        new_df : Pandas DataFrame
            data scaled

        """
        
        x_unscaled = metrics_data[list(metrics_data.columns[:-1])].values # get dataset values (except the label column)
        y = metrics_data["label"]
        
        current_scaler = StandardScaler()
        x = current_scaler.fit_transform(x_unscaled)# train the scaler and transform data
        self.scaler = current_scaler # save the scaler
        new_df = pd.DataFrame(x,columns = list(metrics_data.columns[:-1])) # return to a Pandas DataFrame
        
        
        new_df["label"] = y.values # get back labels
        new_df["id"] = metrics_data.index # same for index
        new_df.set_index("id", drop = True, inplace = True)

        return new_df
    
    def transform(self, metrics_data) :
        """
        scale data(for test dataset) using pretrained the pretrained scaler

        Parameters
        ----------
        metrics_data : Pandas DataFrame
            testing data to scale

        Returns
        -------
        new_df : Pandas DataFrame
            data scaled

        """
        x_unscaled = metrics_data[list(metrics_data.columns[:-1])].values # get dataset values (except the label column)
        y = metrics_data["label"]
        

        x = self.scaler.transform(x_unscaled) # only transform data

        new_df = pd.DataFrame(x,columns = list(metrics_data.columns[:-1])) # return to a Pandas DataFrame
        new_df["label"] = y.values # get back labels
        new_df["id"] = metrics_data.index # same for index
        new_df.set_index("id", drop = True, inplace = True)
        
        return new_df
        
    

def load_and_preprocess_raw_data(directory_data, window_size = 60) :
    """
    process each steps to obtain data in the raw format

    Parameters
    ----------
    directory_data : str
        path to the folder which contains the csv files
    window_size : int, optional
        the size of the window in count of records (not seconds). The default is 60.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    print( "Load datasets...")
    datasets = load_merge_datasets(directory_data)

    splitted_dataset = split_in_windows_raw(datasets, window_size)
    
    
    return adapt_to_dataframe(splitted_dataset)

def split_in_windows_raw(data, window_size):
    """
    split the data in windows of a fixed size for the raw format

    Parameters
    ----------
    data : dic (cf load_merge_datasets)
        the loaded data
    window_size : int
        the size of the window in count of records (not seconds)

    Returns
    -------
    new_data : DataFrame Pandas
        the Dataframe which will be used to train/test models.

    """
    new_data = {}
    
    for key, value in tqdm(data.items()): # for each track
        TBS, label = value[0], value[1] # get TBS values & labels

        if len(TBS)> window_size : # if we have enough data
        
            new_data[key]=(TBS.iloc[0:window_size], label) # we ony keep the first window
        
    return new_data



