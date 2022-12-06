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
    # get the labels columns
    labels = get_labels_columes(subject_df)
    new_subject_df["label"] = subject_df[labels].idxmax(axis=1)
    #add a user_id column in the front
    new_subject_df.insert(0, "user_id", subject_id)
    return new_subject_df

def concatenate_all_subject_df(which_labels: int, is_dominant_hand=True):
    # read all the data from all the subjects
    all_subject_df = pd.DataFrame()
    for subject_id in range(11, 33):
        try:
            subject_df = read_filtered_actigraphy_data_subject(subject_id, which_labels, is_dominant_hand)
            all_subject_df = pd.concat([all_subject_df, subject_df])
        except Exception as e:
            print("Error reading subject {}".format(subject_id))
            print(e)
            continue
    return all_subject_df

# print(read_filtered_actigraphy_data_subject(11, 1, True))

# print("Reading physical activity data")
physical_activity_df = concatenate_all_subject_df(1)
# # print("Reading posture data")
# # posture_df = concatenate_all_subject_df(4)
# # print("Reading behavioral pattern data")
# # behavioral_pattern_df = concatenate_all_subject_df(2)
# # print("Reading high level behavioral pattern data")
# # high_level_behavioral_pattern_df = concatenate_all_subject_df(3)

# # save the dataframes to csv files
physical_activity_df.to_csv(ROOT_DIR + "physical_activity_df.csv", index=False)
# # posture_df.to_csv(ROOT_DIR + "posture_df.csv", index=False)
# # behavioral_pattern_df.to_csv(ROOT_DIR + "behavioral_pattern_df.csv", index=False)
# high_level_behavioral_pattern_df.to_csv(ROOT_DIR + "high_level_behavioral_pattern_df.csv", index=False)
