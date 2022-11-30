import pandas as pd
import numpy as np
import datetime as dt
import os, sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../../.."

def generate_ml_label(night_df, start_wake_time, stop_wake_time):
    # turn the start_wake_time and stop_wake_time to datetime
    start_wake_time = dt.datetime.strptime(start_wake_time, '%Y-%m-%d %H:%M:%S')
    stop_wake_time = dt.datetime.strptime(stop_wake_time, '%Y-%m-%d %H:%M:%S')
    
    # create a new column called DATETIME in night_df and convert the timestamp to datetime
    night_df['DATETIME'] = night_df['timestamp'].apply(lambda x: dt.datetime.fromtimestamp(x))
    #convert to EDT timezone
    night_df['DATETIME'] = night_df['DATETIME'].apply(lambda x: x)
    
    # create a new column called label_wake and set it to 1
    night_df['label_wake'] = 1
    
    # for each DATETIME in night_df, if it is between the start_wake_time and stop_wake_time, set the label_wake to 0
    night_df['label_wake'] = night_df['DATETIME'].apply(lambda x: 0 if x >= start_wake_time and x <= stop_wake_time else 1)
    return night_df

def generate_trainable_data(labeled_df, user_id, is_dominant_hand):
    # create a datafrae to store the trainable data
    trained_df = pd.DataFrame(columns=['timestamp','user_id', 'is_dominant_hand', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 'is_awake'])

    # for each 10 rows in labeled_df, create a new row in trained_df
    for i in range(0, len(labeled_df), 10):
        # get the timestamp of the 10th row
        timestamp = labeled_df.iloc[i+9]['timestamp']
        # user id
        user_id = user_id
        is_dominant_hand = is_dominant_hand
        # get the 10 AUC_sum values
        AUCs = labeled_df.iloc[i:i+10]['AUC_sum']
        
        # get the label_wake of the 10th row
        is_awake = labeled_df.iloc[i+9]['label_wake']
        #create a new row in trained_df
        trained_df.loc[len(trained_df)] = [timestamp, user_id, is_dominant_hand] + AUCs.tolist() + [is_awake]
        
    return trained_df
# for each subject
for subject in range(10, 33):
    print("Processing subject {}".format(subject))
    try:
        # get the wake labels
        wake_label_path = ROOT_DIR + f'/data/PAAWS/HINF_results/wake_labels/DS_{subject}.csv'
        df_wake_labels = pd.read_csv(wake_label_path)
        # get the unique dates in the wake labels
        dates = df_wake_labels['DATE'].unique()
        # for each hand (dominant and non-dominant), create a folder for the hand
        for hand in ["dominant_hand", "non_dominant_hand"]:
            # get the data for each night
            # grab the list of files from the folder
            files = os.listdir(ROOT_DIR + f"/data/PAAWS/HINF_results/AUC_by_night/DS_{subject}/{hand}/")
            # for each file
            for file in files:
                if len(file.split("_")) < 3:
                    continue
                night = file.split("_")[2]
                night = night.replace(".csv", "")
                
                if night in dates:
                    # generate the labels for this night
                    #read the file as pandas dataframe
                    night_df = pd.read_csv(ROOT_DIR + f"/data/PAAWS/HINF_results/AUC_by_night/DS_{subject}/{hand}/{file}")
                    #get the start time and stop time of the wake label
                    start_time = df_wake_labels[df_wake_labels['DATE'] == night]['START_TIME_DATETIME'].values[0]
                    stop_time = df_wake_labels[df_wake_labels['DATE'] == night]['STOP_TIME_DATETIME'].values[0]
                    labeled_night_df = generate_ml_label(night_df, start_time, stop_time)
                    
                    # search for the folder of this subject and hand
                    if not os.path.exists(ROOT_DIR + f"/data/PAAWS/HINF_results/labeled_AUC/DS_{subject}/{hand}/"):
                        os.makedirs(ROOT_DIR + f"/data/PAAWS/HINF_results/labeled_AUC/DS_{subject}/{hand}/")
                        
                    # save the file
                    labeled_night_df.to_csv(ROOT_DIR + f"/data/PAAWS/HINF_results/labeled_AUC/DS_{subject}/{hand}/{night}.csv", index=False)
                    trained_df = generate_trainable_data(labeled_night_df, subject, 1 if hand == "dominant_hand" else 0)
                    
                    # see if the csv file inside the ML_value folder exists
                    try:
                        # read from the csv file
                        current_df = pd.read_csv(ROOT_DIR + f"/data/PAAWS/HINF_results/ML_value/ml.csv")
                        # append the new data to the current dataframe
                        current_df = pd.concat([current_df, trained_df])
                        # save the dataframe to the csv file
                        current_df.to_csv(ROOT_DIR + f"/data/PAAWS/HINF_results/ML_value/ml.csv", index=False)
                    except:
                        # save the dataframe to the csv file
                        trained_df.to_csv(ROOT_DIR + f"/data/PAAWS/HINF_results/ML_value/ml.csv", index=False)
        
    except Exception as e:
            print(f"Error processing subject {subject}".format(subject))
            print(e)
            continue