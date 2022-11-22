import pandas as pd
import numpy as np
import datetime as dt
import os, sys

# get the directory of this file
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../../"

def segment_AUC_into_night(subject_id: int, is_dominant_hand = True):
    if is_dominant_hand:
        file_name = f"DS_{subject_id}_dominant_hand_auc.csv"
    else:
        file_name = f"DS_{subject_id}_nondominant_hand_auc.csv"

    # read the csv file
    df = pd.read_csv(ROOT_DIR + f"data/PAAWS/HINF_results/AUC/{file_name}")
    # turn from datetime to eoch time
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['timestamp'] = df['timestamp'].apply(lambda x: x.timestamp())

    # create a subject folder in the data/PAAWS/HINF_results/AUC_by_night folder
    #check if the folder exists
    if not os.path.exists(ROOT_DIR + f"data/PAAWS/HINF_results/AUC_by_night/DS_{subject_id}"):
        os.makedirs(ROOT_DIR + f"data/PAAWS/HINF_results/AUC_by_night/DS_{subject_id}")
    
    #create a dominant_hand folder or non_dominant_hand folder
    if is_dominant_hand:
        if not os.path.exists(ROOT_DIR + f"data/PAAWS/HINF_results/AUC_by_night/DS_{subject_id}/dominant_hand"):
            os.makedirs(ROOT_DIR + f"data/PAAWS/HINF_results/AUC_by_night/DS_{subject_id}/dominant_hand")
    else:
        if not os.path.exists(ROOT_DIR + f"data/PAAWS/HINF_results/AUC_by_night/DS_{subject_id}/non_dominant_hand"):
            os.makedirs(ROOT_DIR + f"data/PAAWS/HINF_results/AUC_by_night/DS_{subject_id}/non_dominant_hand")
    # from the timestamp column, get the date
    df["date"] = df["timestamp"].apply(lambda x: dt.datetime.fromtimestamp(x).date())
    #ge the unique dates
    dates = df["date"].unique()
    # for each date, create a csv file with the AUC values for that date
    for date in dates:
        # get data from the previous day
        previous_day = date - dt.timedelta(days = 1)
        df_previous_day = df[df["date"] == previous_day]
        # print length of the previous day
        # print length of the current day
        df_date = df[df["date"] == date]
        #remove the date column
        df_date = df_date.drop(columns=["date"])
        df_previous_day = df_previous_day.drop(columns=["date"])
        # get the hour of the day
        df_date["hour"] = df_date["timestamp"].apply(lambda x: dt.datetime.fromtimestamp(x).hour)
        df_previous_day["hour"] = df_previous_day["timestamp"].apply(lambda x: dt.datetime.fromtimestamp(x).hour)
        # only get the data from 10pm to midnight of the previous day
        df_previous_day = df_previous_day[(df_previous_day["hour"] >= 22)]
        # only get the data from midnight to 10am of the current day
        df_date = df_date[(df_date["hour"] >= 0) & (df_date["hour"] <= 10)]
        # concatenate the two dataframes
        df_night = pd.concat([df_previous_day, df_date])
        # reinde the dataframe's index
        df_night = df_night.reset_index(drop=True)
        # sort the dataframe by timestamp
        df_night = df_night.sort_values(by=["timestamp"])
        # remove the hour column
        df_night = df_night.drop(columns=["hour"])
        # print the length of the dataframe
        print("Length of the dataframe: ", len(df_night))
        
        # save the AUC values for that date
        if is_dominant_hand:
            df_night.to_csv(ROOT_DIR + f"data/PAAWS/HINF_results/AUC_by_night/DS_{subject_id}/dominant_hand/auc_night_{date}.csv", index=False)
        else:
            df_night.to_csv(ROOT_DIR + f"data/PAAWS/HINF_results/AUC_by_night/DS_{subject_id}/non_dominant_hand/auc_night_{date}.csv", index=False)

for i in range(10, 33):
    print(f"Processing subject {i}")
    # segment_AUC_into_night(i, is_dominant_hand=False)
    try:
        segment_AUC_into_night(i, is_dominant_hand=True)
    except Exception as e:
        print(f"Error processing subject {i} dominant hand")
        print(e)
    
    try:
        segment_AUC_into_night(i, is_dominant_hand=False)
    except:
        print(f"Error processing subject {i} non dominant hand")
