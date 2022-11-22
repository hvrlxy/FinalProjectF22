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
    file_path = ROOT_DIR + f"data/PAAWS/filtered/DS_{subject_id}_nondominant_hand_accel_data.csv"
    # read the csv file
    df = pd.read_csv(file_path)
    # turn timestamp into datetime to milliseconds
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    # rename the columns to x, y, z, timestamp
    df = df.rename(columns={"Accelerometer X": "x", "Accelerometer Y": "y", "Accelerometer Z": "z"})
    # resampling the data to 50Hz
    df = df.resample("20ms", on="timestamp").mean()
    return df 

def compute_AUC(df, window_size=50 * 10):
    """
    Compute the moving average of the data
    """
    # compute the moving average for each axis
    df['MA_x'] = df['x'].rolling(window_size).mean()
    df['MA_y'] = df['y'].rolling(window_size).mean()
    df['MA_z'] = df['z'].rolling(window_size).mean()
    # return the dataframe with only timestamp and auc_sum
    df['AUC_x'] = df['x'] - df['MA_x']
    df['AUC_y'] = df['y'] - df['MA_y']
    df['AUC_z'] = df['z'] - df['MA_z']
    # get the absolute value of the AUC
    df['AUC_x'] = df['AUC_x'].abs()
    df['AUC_y'] = df['AUC_y'].abs()
    df['AUC_z'] = df['AUC_z'].abs()
    # for each 500 data points, compute the sum of AUC_x, AUC_y, AUC_z
    df['AUC_sum'] = df['AUC_x'] + df['AUC_y'] + df['AUC_z']
    temp_df = pd.DataFrame({'timestamp': df.index, 'AUC_sum': df['AUC_sum']})
    # reindex the data frame to 1, 2 ...
    temp_df = temp_df.reset_index(drop=True)

    ten_seconds_auc_series = temp_df['AUC_sum'].groupby(temp_df.index // 500).sum()

    ten_seconds_timestamp_series = temp_df['timestamp'].groupby(temp_df.index // 500).first()
    new_df = pd.DataFrame({'timestamp': ten_seconds_timestamp_series, 'AUC_sum': ten_seconds_auc_series})
    return new_df

for i in range(10, 33):
    print(f"Processing subject {i}")
    try:
        df = read_data(i)
        new_df = compute_AUC(df)
        new_df.to_csv(ROOT_DIR + f"data/PAAWS/HINF_results/AUC/DS_{i}_nondominant_hand_auc.csv", index=False)
    except:
        print(f"Error processing subject {i}")