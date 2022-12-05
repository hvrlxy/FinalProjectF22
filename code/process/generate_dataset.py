import os, sys
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import sklearn.preprocessing as preprocessing
from generate_labels_summary import get_labels_synced_actigraphy_file, get_labels_columes
import warnings

# ignore the warning
warnings.filterwarnings("ignore")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../../"
print(ROOT_DIR)
labels_type = {1: "PhysicalActivity",
                2: "BehavioralPattern",
                3: "HighLevelBehavioralPattern",
                4: "Posture"}

def read_filtered_actigraphy_data_subject(subject_id: int, which_labels: int, is_dominant_hand=True):
    subject_df = get_labels_synced_actigraphy_file(subject_id, which_labels, is_dominant_hand=True)
    # remove all the labels columes that do not contains stand, sit, staitr, cycling, sit, lying, run
    #first get the timestamp, and the features columns
    cols = subject_df.columns
    # print(cols)
    original_cols = [col for col in cols if col[0] == "x" or col[0] == "y" or col[0] == "z" or col == "timestamp"]
    # print(original_cols)
    new_subject_df = subject_df[original_cols]
    final_labels = []
    for labels in get_labels_columes(subject_df):
        # if labels.lower() contains any of the following words, add it to the new df
        kept_labels = ["stand", "sit", "stair", "cycling", "sit", "lying", "run"]
        for words in kept_labels:
            if words in labels.lower():
                new_subject_df[labels] = subject_df[labels]
                final_labels.append(labels)
                break

    # remove all the rows that do not contain any labels
    new_subject_df = new_subject_df.dropna(axis=0, how="all", subset=get_labels_columes(new_subject_df))
    
    #create a dataframe which only contains the timestamp and features columns
    final_subject_df = pd.DataFrame()
    for label in final_labels:
        label_df = new_subject_df[[col for col in new_subject_df.columns if col not in final_labels]]
        label_df[label] = new_subject_df[label]
        #drop all the row with value in the label colume is not 1
        label_df = label_df[label_df[label] == 1]
        # drop all other labels colume except the one we are working on
        label_df = label_df[[col for col in label_df.columns if col not in final_labels or col == label]]
        # add a class column and set it to the label
        label_df["class"] = label
        # drop label colume
        label_df = label_df[[col for col in label_df.columns if col != label]]
        #  add label df to the final df
        final_subject_df = final_subject_df.append(label_df)
        
    # combining labels with stair in the name in final_subject-df
    final_subject_df["class"] = final_subject_df["class"].apply(lambda x: "STAIR" if "stair" in x.lower() else x)
    # combining labels with stand in the name in final_subject-df
    final_subject_df["class"] = final_subject_df["class"].apply(lambda x: "STILL" if "still" in x.lower() else x)
    
    # add one columes indicating the hour of the timestamp, in EDT time
    #first, we must convert timestamp to datetime (in EDT time)
    final_subject_df["hour"] = pd.to_datetime(final_subject_df["timestamp"], unit="s")
    final_subject_df["hour"] = final_subject_df["hour"].apply(lambda x: x.hour)
    # put the hour colume in the front
    cols = final_subject_df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    final_subject_df = final_subject_df[cols]
    
    return final_subject_df

def concatenate_all_subject_df(which_labels: int, is_dominant_hand=True):
    # read all the data from all the subjects
    all_subject_df = pd.DataFrame()
    for subject_id in range(10, 33):
        try:
            subject_df = read_filtered_actigraphy_data_subject(subject_id, which_labels, is_dominant_hand)
            all_subject_df = all_subject_df.append(subject_df)
        except:
            print("Error reading subject {}".format(subject_id))
            continue
    return all_subject_df

print("Reading physical activity data")
physical_activity_df = concatenate_all_subject_df(1)
print("Reading posture data")
posture_df = concatenate_all_subject_df(4)
print("Reading behavioral pattern data")
behavioral_pattern_df = concatenate_all_subject_df(2)
print("Reading high level behavioral pattern data")
high_level_behavioral_pattern_df = concatenate_all_subject_df(3)

# save the dataframes to csv files
physical_activity_df.to_csv(ROOT_DIR + "physical_activity_df.csv", index=False)
posture_df.to_csv(ROOT_DIR + "posture_df.csv", index=False)
behavioral_pattern_df.to_csv(ROOT_DIR + "behavioral_pattern_df.csv", index=False)
high_level_behavioral_pattern_df.to_csv(ROOT_DIR + "high_level_behavioral_pattern_df.csv", index=False)
