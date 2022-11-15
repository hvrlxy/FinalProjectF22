import os, sys
import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import datetime

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../"

def get_file_name(subject_id: int, path: str):
    '''
    This function takes in the subject's id and return the list of timestamp-synced actigraph files for the subject
    ::params:subject_id: id of the subject
    '''
    folder_path =  path + "data/actigraphy_features/"
    files_lst = os.listdir(folder_path)
    #find if any files named DS_subject_id is in the folder
    subject_files = [file for file in files_lst if f'DS_{subject_id}' in file]
    print(f"Files found for subject {subject_id}: ", subject_files)
    return subject_files

def get_filtered_behavior_pattern_labels(subject_id):
    '''
    This function takes the subject_id and return the dataframe for the behavioral pattern labels
    '''
    # read in the behavioral pattern labels
    behavior_pattern_labels_df = pd.read_csv(f"{ROOT_DIR}/data/filtered_labels/BehavioralPattern/DS_{subject_id}_combined_BehavioralParameters_corr.csv")
    # get all the columns that are not timestamp
    behavior_pattern_labels = [col for col in behavior_pattern_labels_df.columns if col != "START_TIME" and col != "STOP_TIME"]
    return behavior_pattern_labels, behavior_pattern_labels_df

def get_filtered_physical_activity_labels(subject_id):
    '''
    This function takes the subject_id and return the dataframe for the activity labels
    '''
    # read in the activity labels
    activity_labels_df = pd.read_csv(f"{ROOT_DIR}/data/filtered_labels/PhysicalActivity/DS_{subject_id}_combined_PhysicalActivity.csv")
    # get all the columns that are not timestamp
    activity_labels = [col for col in activity_labels_df.columns if col != "START_TIME" and col != "STOP_TIME"]
    return activity_labels, activity_labels_df

def get_filtered_high_level_behavioral_pattern(subject_id):
    '''
    This function takes the subject_id and return the dataframe for the high level behavioral pattern labels
    '''
    # read in the high level behavioral pattern labels
    high_level_behavioral_pattern_df = pd.read_csv(f"{ROOT_DIR}/data/filtered_labels/HighLevelBehavioralPattern/DS_{subject_id}_combined_HighLevelBehavioralPatterns.csv")
    # get all the columns that are not timestamp
    high_level_behavioral_pattern = [col for col in high_level_behavioral_pattern_df.columns if col != "START_TIME" and col != "STOP_TIME"]
    return high_level_behavioral_pattern, high_level_behavioral_pattern_df

def get_filtered_posture_labels(subject_id):
    '''
    This function takes the subject_id and return the dataframe for the posture labels
    '''
    # read in the posture labels
    posture_labels_df = pd.read_csv(f"{ROOT_DIR}/data/filtered_labels/Posture/DS_{subject_id}_combined_Posture.csv")
    # get all the columns that are not timestamp
    posture_labels = [col for col in posture_labels_df.columns if col != "START_TIME" and col != "STOP_TIME"]
    return posture_labels, posture_labels_df

def sync_physical_activity_labels(subject_id, is_dominant_hand = True):
    # get the list of actigraph files for the subject
    subject_files = get_file_name(subject_id, ROOT_DIR)
    # get the df of physical activity labels
    activity_labels, activity_labels_df = get_filtered_physical_activity_labels(subject_id)
    #convert the start and stop time to epoch time
    activity_labels_df["START_TIME"] = pd.to_datetime(activity_labels_df["START_TIME"])
    activity_labels_df["STOP_TIME"] = pd.to_datetime(activity_labels_df["STOP_TIME"])
    # plus 5 hours to convert to UTC time
    activity_labels_df["START_TIME"] = activity_labels_df["START_TIME"] + datetime.timedelta(hours=5)
    activity_labels_df["STOP_TIME"] = activity_labels_df["STOP_TIME"] + datetime.timedelta(hours=5)
    # convert to epoch time using the timestamp function()
    activity_labels_df["START_TIME"] = activity_labels_df["START_TIME"].apply(lambda x: x.timestamp())
    activity_labels_df["STOP_TIME"] = activity_labels_df["STOP_TIME"].apply(lambda x: x.timestamp())
    if is_dominant_hand:
        # get the df of actigraph data from the dominant wrist
        actigraph_df = pd.read_csv(f"{ROOT_DIR}/data/actigraphy_features/{subject_files[0]}")
    else:
        # get the df of actigraph data from the non-dominant wrist
        actigraph_df = pd.read_csv(f"{ROOT_DIR}/data/actigraphy_features/{subject_files[1]}")
    # create one column for each of the activity labels in the activity_labels_df to the actigraph_df
    for label in activity_labels:
        actigraph_df[label] = 0
    # iterate through the activity_labels_df and add the label to the corresponding timestamp in the actigraph_df
    for index, row in activity_labels_df.iterrows():
        # get the start and end time of the activity
        start_time = row["START_TIME"]
        end_time = row["STOP_TIME"]
        # ampped the entries in the label columns to the corresponding timestamp between start_time and end_time in the actigraph_df
        actigraph_df.loc[(actigraph_df["timestamp"] >= start_time) & (actigraph_df["timestamp"] <= end_time), activity_labels] = row[activity_labels].values
    if is_dominant_hand:
            # save the actigraph_df to a csv file
        actigraph_df.to_csv(f"{ROOT_DIR}/data/labeled_actigraph/PhysicalActivity/DS_{subject_id}_combined_PhysicalActivity_dominant_synced.csv", index=False)
    else:
        # save the actigraph_df to a csv file
        actigraph_df.to_csv(f"{ROOT_DIR}/data/labeled_actigraph/PhysicalActivity/DS_{subject_id}_combined_PhysicalActivity_non_dominant_synced.csv", index=False)

def sync_behavioral_pattern_labels(subject_id, is_dominant_hand = True):
    # get the list of actigraph files for the subject
    subject_files = get_file_name(subject_id, ROOT_DIR)
    # get the df of behavioral pattern labels
    behavior_pattern_labels, behavior_pattern_labels_df = get_filtered_behavior_pattern_labels(subject_id)
    #convert the start and stop time to epoch time
    behavior_pattern_labels_df["START_TIME"] = pd.to_datetime(behavior_pattern_labels_df["START_TIME"])
    behavior_pattern_labels_df["STOP_TIME"] = pd.to_datetime(behavior_pattern_labels_df["STOP_TIME"])
    # plus 5 hours to the start and stop time to convert it to UTC time
    behavior_pattern_labels_df["START_TIME"] = behavior_pattern_labels_df["START_TIME"] + pd.Timedelta(hours=5)
    behavior_pattern_labels_df["STOP_TIME"] = behavior_pattern_labels_df["STOP_TIME"] + pd.Timedelta(hours=5)
    # convert to epoch time using timestamp()
    behavior_pattern_labels_df["START_TIME"] = behavior_pattern_labels_df["START_TIME"].apply(lambda x: x.timestamp())
    behavior_pattern_labels_df["STOP_TIME"] = behavior_pattern_labels_df["STOP_TIME"].apply(lambda x: x.timestamp())
    if is_dominant_hand:
        # get the df of actigraph data from the dominant wrist
        actigraph_df = pd.read_csv(f"{ROOT_DIR}/data/actigraphy_features/{subject_files[0]}")
    else:
        # get the df of actigraph data from the non-dominant wrist
        actigraph_df = pd.read_csv(f"{ROOT_DIR}/data/actigraphy_features/{subject_files[1]}")
    # create one column for each of the activity labels in the activity_labels_df to the actigraph_df
    for label in behavior_pattern_labels:
        actigraph_df[label] = 0
    # iterate through the activity_labels_df and add the label to the corresponding timestamp in the actigraph_df
    for index, row in behavior_pattern_labels_df.iterrows():
        # get the start and end time of the activity
        start_time = row["START_TIME"]
        end_time = row["STOP_TIME"]
        # ampped the entries in the label columns to the corresponding timestamp between start_time and end_time in the actigraph_df
        actigraph_df.loc[(actigraph_df["timestamp"] >= start_time) & (actigraph_df["timestamp"] <= end_time), behavior_pattern_labels] = row[behavior_pattern_labels].values
    if is_dominant_hand:
            # save the actigraph_df to a csv file
        actigraph_df.to_csv(f"{ROOT_DIR}/data/labeled_actigraph/BehavioralPattern/DS_{subject_id}_combined_BehavioralParameters_corr_dominant_synced.csv", index=False)
    else:
        # save the actigraph_df to a csv file
        actigraph_df.to_csv(f"{ROOT_DIR}/data/labeled_actigraph/BehavioralPattern/DS_{subject_id}_combined_BehavioralParameters_corr_non_dominant_synced.csv", index=False)
    
def sync_high_level_labels(subject_id, is_dominant_hand = True):
    # get the list of actigraph files for the subject
    subject_files = get_file_name(subject_id, ROOT_DIR)
    # get the df of high level labels
    high_level_labels, high_level_labels_df = get_filtered_high_level_behavioral_pattern(subject_id)
    #convert the start and stop time to epoch time
    high_level_labels_df["START_TIME"] = pd.to_datetime(high_level_labels_df["START_TIME"])
    high_level_labels_df["STOP_TIME"] = pd.to_datetime(high_level_labels_df["STOP_TIME"])
    # plus 5 hours to the start and stop time to convert it to UTC time
    high_level_labels_df["START_TIME"] = high_level_labels_df["START_TIME"] + pd.Timedelta(hours=5)
    high_level_labels_df["STOP_TIME"] = high_level_labels_df["STOP_TIME"] + pd.Timedelta(hours=5)
    # convert the start and stop time to epoch time using timestamp()
    high_level_labels_df["START_TIME"] = high_level_labels_df["START_TIME"].apply(lambda x: x.timestamp())
    high_level_labels_df["STOP_TIME"] = high_level_labels_df["STOP_TIME"].apply(lambda x: x.timestamp())
    if is_dominant_hand:
        # get the df of actigraph data from the dominant wrist
        actigraph_df = pd.read_csv(f"{ROOT_DIR}/data/actigraphy_features/{subject_files[0]}")
    else:
        # get the df of actigraph data from the non-dominant wrist
        actigraph_df = pd.read_csv(f"{ROOT_DIR}/data/actigraphy_features/{subject_files[1]}")
    # create one column for each of the activity labels in the activity_labels_df to the actigraph_df
    for label in high_level_labels:
        actigraph_df[label] = 0
    # iterate through the activity_labels_df and add the label to the corresponding timestamp in the actigraph_df
    for index, row in high_level_labels_df.iterrows():
        # get the start and end time of the activity
        start_time = row["START_TIME"]
        end_time = row["STOP_TIME"]
        # ampped the entries in the label columns to the corresponding timestamp between start_time and end_time in the actigraph_df
        actigraph_df.loc[(actigraph_df["timestamp"] >= start_time) & (actigraph_df["timestamp"] <= end_time), high_level_labels] = row[high_level_labels].values
    if is_dominant_hand:
            # save the actigraph_df to a csv file
        actigraph_df.to_csv(f"{ROOT_DIR}/data/labeled_actigraph/HighLevelBehavioralPattern/DS_{subject_id}_combined_HighLevel_dominant_synced.csv", index=False)
    else:
        # save the actigraph_df to a csv file
        actigraph_df.to_csv(f"{ROOT_DIR}/data/labeled_actigraph/HighLevelBehavioralPattern/DS_{subject_id}_combined_HighLevel_non_dominant_synced.csv", index=False)

def sync_posture_labels(subject_id, is_dominant_hand = True):
    # get the list of actigraph files for the subject
    subject_files = get_file_name(subject_id, ROOT_DIR)
    # get the df of posture labels
    posture_labels, posture_labels_df = get_filtered_posture_labels(subject_id)
    #convert the start and stop time to epoch time given that the time zone is EDT
    posture_labels_df["START_TIME"] = pd.to_datetime(posture_labels_df["START_TIME"])
    posture_labels_df["STOP_TIME"] = pd.to_datetime(posture_labels_df["STOP_TIME"])
    # minus 5 hours to convert the time to UTC
    posture_labels_df["START_TIME"] = posture_labels_df["START_TIME"] + pd.Timedelta(hours=5)
    posture_labels_df["STOP_TIME"] = posture_labels_df["STOP_TIME"] + pd.Timedelta(hours=5)
    # print("Start time posture before: \n", list(posture_labels_df['START_TIME'])[:5])
    #  apply the timestamp() function to the start and stop time columns
    posture_labels_df["START_TIME"] = posture_labels_df["START_TIME"].apply(lambda x: x.timestamp())
    posture_labels_df["STOP_TIME"] = posture_labels_df["STOP_TIME"].apply(lambda x: x.timestamp())
    # print("Start time posture after: \n", list(posture_labels_df['START_TIME'])[:5])
    if is_dominant_hand:
        # get the df of actigraph data from the dominant wrist
        actigraph_df = pd.read_csv(f"{ROOT_DIR}/data/actigraphy_features/{subject_files[0]}")
    else:
        # get the df of actigraph data from the non-dominant wrist
        actigraph_df = pd.read_csv(f"{ROOT_DIR}/data/actigraphy_features/{subject_files[1]}")
    # create one column for each of the activity labels in the activity_labels_df to the actigraph_df
    for label in posture_labels:
        actigraph_df[label] = 0
    # iterate through the activity_labels_df and add the label to the corresponding timestamp in the actigraph_df
    for index, row in posture_labels_df.iterrows():
        # get the start and end time of the activity
        start_time = row["START_TIME"]
        end_time = row["STOP_TIME"]
        # ampped the entries in the label columns to the corresponding timestamp between start_time and end_time in the actigraph_df
        actigraph_df.loc[(actigraph_df["timestamp"] >= start_time) & (actigraph_df["timestamp"] <= end_time), posture_labels] = row[posture_labels].values
    if is_dominant_hand:
            # save the actigraph_df to a csv file
        actigraph_df.to_csv(f"{ROOT_DIR}/data/labeled_actigraph/Posture/DS_{subject_id}_combined_Posture_dominant_synced.csv", index=False)
    else:
        # save the actigraph_df to a csv file
        actigraph_df.to_csv(f"{ROOT_DIR}/data/labeled_actigraph/Posture/DS_{subject_id}_combined_Posture_non_dominant_synced.csv", index=False)

for i in range(15, 33):
    sync_physical_activity_labels(i, is_dominant_hand = True)
    sync_behavioral_pattern_labels(i, is_dominant_hand = True)
    sync_high_level_labels(i, is_dominant_hand = True)
    sync_posture_labels(i, is_dominant_hand = True)
