import pandas as pd
import numpy as np
import datetime as dt
import os, sys
import warnings
warnings.filterwarnings("ignore")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../../../"

def get_actigraphy_features(subject: int):
    # get the csv files with the features for the subject
    actigraphy_features = pd.read_csv(ROOT_DIR + f"data/PAAWS/actigraphy_features/DS_{subject}_actigraphy_features_dominant.csv")
    # get the wake labels for the subject
    wake_labels = pd.read_csv(ROOT_DIR + f"data/PAAWS/HINF_results/wake_labels/DS_{subject}.csv")

    # append datetime column to the actigraphy features
    actigraphy_features['datetime'] = actigraphy_features['timestamp'].apply(lambda x: dt.datetime.fromtimestamp(x))
    actigraphy_features['timestamp'] = actigraphy_features['timestamp'].astype(int)
    # initalize the final dataframe
    final_df = pd.DataFrame()

    # for each row in the wake labels, get the start time and stop time
    for index, row in wake_labels.iterrows():
        start_time_sleep = row['START_TIME_DATETIME']
        stop_time_sleep = row['STOP_TIME_DATETIME']
        # grab the data between 2 hours before the start time and 2 hours after the stop time
        start_time = dt.datetime.strptime(start_time_sleep, '%Y-%m-%d %H:%M:%S') - dt.timedelta(hours=2)
        stop_time = dt.datetime.strptime(stop_time_sleep, '%Y-%m-%d %H:%M:%S') + dt.timedelta(hours=2)
        # convert the start time and stop time to epoch time in seconds
        start_time = int(start_time.timestamp())
        stop_time = int(stop_time.timestamp())
        # print(start_time_sleep, stop_time_sleep, start_time, stop_time)
        
        # filter out the actigraphy features and only take the one between the start time and stop time
        df = actigraphy_features[(actigraphy_features['timestamp'] >= start_time)]
        df = df[(df['timestamp'] <= stop_time)]
        # print(start_time_sleep, stop_time_sleep, df['datetime'].to_list()[0], df['datetime'].to_list()[-1])
        # add the label column
        # for the time between start_time_sleep and stop_time_sleep, the label is 1
        # for the time before start_time_sleep and after stop_time_sleep, the label is 0
        df['is_awake'] = np.where((df['datetime'] >= start_time_sleep) & (df['datetime'] <= stop_time_sleep), 0, 1)
        # add the subject column
        df['subject'] = subject
        # add the data to the final dataframe
        final_df = pd.concat([final_df, df])
    return final_df

def generate_ml_data():
    # initalize the final dataframe
    final_df = pd.DataFrame()
    # for each subject, get the actigraphy features
    for subject in range(10, 33):
        try:
            df = get_actigraphy_features(subject)
            # add the data to the final dataframe
            final_df = pd.concat([final_df, df])
        except:
            print(f"Subject {subject} does not have actigraphy features")
    # save the final dataframe to a csv file
    final_df.to_csv(ROOT_DIR + f"data/PAAWS/HINF_results/ML_value/wake_features.csv", index=False)

generate_ml_data()
