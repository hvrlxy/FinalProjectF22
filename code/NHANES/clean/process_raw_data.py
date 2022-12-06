import os, sys, subprocess
import pandas as pd
import datetime as dt
import numpy as np
import scipy.stats as stats
from feature_extract import ActigraphSummary
import gzip
file_lst = ['65704.csv.gz', '69247.csv.gz', '80458.csv.gz']

def convert_string_to_datetime(start_time: str, start_date: str):
    '''
    Converts the start time and start date to a datetime object.
    :param start_time: the start time in string format
    :param start_date: the start date in string format
    :return: the start time in datetime format
    '''
    # print(start_time.decode("utf-8"), start_date.decode("utf-8"))
    # from start_time and start_date, get the start time in date time format
    dt_format = dt.datetime.strptime(start_date, "%m/%d/%Y")
    # convert the start_time to a timedelta object
    start_time = dt.datetime.strptime(start_time, "%H:%M:%S")
    start_time = dt.timedelta(hours=start_time.hour, minutes=start_time.minute, seconds=start_time.second)
    # add the start_time to the start_date
    start_time = dt_format + start_time
    return start_time

def read_raw_accel_data(file_name: int):
    '''
    Reads the raw dominant hand acceleration data from the csv file.
    '''
    # remove the first 10 lines of the csv file
    df = pd.read_csv(file_name, skiprows=10)

    # get the third and fourth line from the text file
    with gzip.open(file_name, "rt") as f:
        lines = f.readlines()
        start_time = lines[2]
        start_date = lines[3]
        #remove Start Time: and Start Date: from the lines
        start_date = str(start_date.replace("Start Date ", "").replace("\n", ""))
        start_time = str(start_time.replace("Start Time ", "").replace("\n", ""))
        # from start_time and start_date, get the start time in date time format
        start_time = convert_string_to_datetime(start_time, start_date)
        # convert start_time into epoch time
        start_time = start_time.timestamp()
    # get the epoch value of 1/80 of a second
    epoch = 1/80
    #add the timestamp column to the dataframe starting from the start_time and incrementing by 1/80th of a second
    timstamp_lst = [start_time + epoch * i for i in range(len(df))]
    # set timestamp as the first column
    df.insert(0, "timestamp", timstamp_lst)
    return df

def extract_and_save_features(file_name):
    file_name = '/Users/hale/Desktop/NEU-CLASS/FinalProjectF22/data/NHANES/raw/' + file_name
    raw_df = read_raw_accel_data(file_name)
    act_sum = ActigraphSummary(raw_df)
    feature_df = act_sum.segment_and_add_features()
    
    # save the feature_df to a csv file
    feature_df.to_csv("/Users/hale/Desktop/NEU-CLASS/FinalProjectF22/data/NHANES/features_extracted/" + file_name, index=False)
    
for file in file_lst:
    extract_and_save_features(file)