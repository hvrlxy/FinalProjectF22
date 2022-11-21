import pandas as pd
import numpy as np
import os, sys

# get the root directory of the project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../../"

def read_data(subject_id):
    """
    Read the data from the csv file
    """
    # get the path of the csv file
    # file_path = ROOT_DIR + f"data/filtered_labels/{label_type}/DS_{subject_id}_combined_{label_type}.csv"
    file_path = ROOT_DIR + f"data/zip/DS_{subject_id}_dominant_hand_accel_data.csv"
    # read the csv file
    df = pd.read_csv(file_path)
    # turn timestamp into datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # rename the columns to x, y, z, timestamp
    df = df.rename(columns={"Accelerometer X": "x", "Accelerometer Y": "y", "Accelerometer Z": "z"})
    # resampling the data to 50Hz
    df = df.resample("20ms", on="timestamp").mean()
    return df 

def compute_moving_average(df, window_size=50 * 10):
    """
    Compute the moving average of the data
    """
    # compute the moving average for each axis
    df['x'] = df['x'].rolling(window_size).mean()
    df['y'] = df['y'].rolling(window_size).mean()
    df['z'] = df['z'].rolling(window_size).mean()
    df['auc_sum'] = df['x'] + df['y'] + df['z']
    
    # return the dataframe with only timestamp and auc_sum
    new_df = pd.DataFrame()
    new_df['timestamp'] = df.index
    new_df['auc_sum'] = df['auc_sum']
    return new_df

df = read_data(10)
print(df.head())
auc_df = compute_moving_average(df)
# save the data to csv file
auc_df.to_csv(ROOT_DIR + f"data/zip/DS_{10}_auc.csv", index=False)