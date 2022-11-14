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

def get_filtered__behavior_pattern_labels(subject_id):
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
    activity_labels_df["START_TIME"] = activity_labels_df["START_TIME"].astype(np.int64) // 10**9
    activity_labels_df["STOP_TIME"] = activity_labels_df["STOP_TIME"].astype(np.int64) // 10**9
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
        actigraph_df.to_csv(f"{ROOT_DIR}/data/labeled_actigraph/PhysicalActivity/DS_{subject_id}_combined_PhysicalActivity_synced.csv", index=False)

#test the function
# behavior_pattern_labels, behavior_pattern_labels_df = get_filtered__behavior_pattern_labels(10)
# activity_labels, activity_labels_df = get_filtered_physical_activity_labels(10)
# high_level_behavioral_pattern, high_level_behavioral_pattern_df = get_filtered_high_level_behavioral_pattern(10)
# posture_labels, posture_labels_df = get_filtered_posture_labels(10)
# print(behavior_pattern_labels)
# print(behavior_pattern_labels_df.head())
# print(activity_labels)
# print(activity_labels_df.head())
# print(high_level_behavioral_pattern)
# print(high_level_behavioral_pattern_df.head())
# print(posture_labels)
# print(posture_labels_df.head())
# for i in range(23, 33):
#     sync_physical_activity_labels(i)

# sync_physical_activity_labels(23, is_dominant_hand = True)
