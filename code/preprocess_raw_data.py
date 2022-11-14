import os, sys, subprocess
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# get the directory of this file
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../" 

def get_file_name(subject_id: int):
    '''
    Returns the file name of the csv file for the given subject and hand.
    '''
    folder = ROOT_DIR + f"data/raw/DS-{subject_id}/"
    # get all the files in the folder
    files = os.listdir(folder)
    # print(files) 
    # get the file name of the dominant hand
    if len([file for file in files if "DominantWrist" in file or "LeftWrist" in file or "Dominantwrist" in file]) > 0:
        dominant_file_name = [file for file in files if "DominantWrist" in file or "LeftWrist" in file or "Dominantwrist" in file][0]
    else:
        dominant_file_name = ""
    # get the file name of the non-dominant hand
    if len([file for file in files if "NondominantWrist" in file or "RightWrist" in file or "Nondominantwrist" in file]) > 0:
        nondominant_file_name = [file for file in files if "NondominantWrist" in file or "RightWrist" in file or "Nondominantwrist" in file][0]
    else:
        nondominant_file_name = ""
        return folder + dominant_file_name, None
    return folder + dominant_file_name, folder + nondominant_file_name

def convert_string_to_datetime(start_time: str, start_date: str):
    '''
    Converts the start time and start date to a datetime object.
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

def read_raw_dominant_hand_accel_data(subject_id: int):
    '''
    Reads the raw dominant hand acceleration data from the csv file.
    '''
    # get the file name of the dominant hand csv file
    dominant_file_name = get_file_name(subject_id)[0]
    # remove the first 10 lines of the csv file
    df = pd.read_csv(dominant_file_name, skiprows=10)

    # get the third and fourth line from the text file
    with open(dominant_file_name, "r") as f:
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
    # save the dataframe to a csv file
    df.to_csv(ROOT_DIR + f"data/filtered/DS_{subject_id}_dominant_hand_accel_data.csv", index=False)

def read_raw_nondominant_hand_accel_data(subject_id: int):
    '''
    Reads the raw non-dominant hand acceleration data from the csv file.
    '''
    # get the file name of the dominant hand csv file
    nondominant_file_name = get_file_name(subject_id)[1]

    #check if the file name is empty
    if nondominant_file_name == "":
        print("No non-dominant file found for subject: ", subject_id)
        return
    # remove the first 10 lines of the csv file
    df = pd.read_csv(nondominant_file_name, skiprows=10)

    # get the third and fourth line from the text file
    with open(nondominant_file_name, "r") as f:
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
    # save the dataframe to a csv file
    df.to_csv(ROOT_DIR + f"data/filtered/DS_{subject_id}_nondominant_hand_accel_data.csv", index=False)

for i in range(10, 33):
    try:
        read_raw_dominant_hand_accel_data(i)
        # print some information about the file
        print(f"Subject {i} - dominant hand done")
        read_raw_nondominant_hand_accel_data(i)
        # print some information about the file
        print(f"Subject {i} - non-dominant hand done")
    except Exception as e:
    #     # print the error
        print(e)
        print(f"Error with subject {i}")

