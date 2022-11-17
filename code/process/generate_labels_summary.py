import os, sys 
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# get the directory of this file
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../../"
print("Root directory: ", ROOT_DIR)

#  initialize a dictionary to store the labels type
labels_type = {1: "PhysicalActivity",
                2: "BehavioralPattern",
                3: "HighLevelBehavioralPattern",
                4: "Posture",
                "PhysicalActivity": 1,
                "BehavioralPattern": 2,
                "HighLevelBehavioralPattern": 3,
                "Posture": 4}

def get_labels_synced_actigraphy_file(subject_id, which_labels: int, is_dominant_hand=True):
    # get the list of files inside the labeled_actigraphy folder
    folder = ROOT_DIR + f"data/PAAWS/labeled_actigraph/{labels_type[which_labels]}/"
    files = os.listdir(folder)
    if is_dominant_hand:
        try:
            file_name = [file for file in files if f"DS_{subject_id}" in file][0]
        except:
            print("Dominant hand file not found")
            return None
    else:
        try:
            file_name = [file for file in files if f"DS_{subject_id}" in file][1]
        except:
            print("Non-dominant hand file not found")
            return None

    # read the csv file
    df = pd.read_csv(folder + file_name)
    return df

def get_labels_columes(df):
    # get the columns of the labels
    columns = df.columns
    # the colmes with the labels is the one that does not start with x, y, z or timestamp
    columns = [col for col in columns if col[0] != "x" and col[0] != "y" and col[0] != "z" and col != "timestamp"]
    return columns

def get_labels_summary(subject_id: int, which_labels: int, is_dominant_hand=True):
    # get the df with the labels
    actigraph_df = get_labels_synced_actigraphy_file(subject_id, which_labels, is_dominant_hand=True)
    if actigraph_df is None:
        # if we cannot find the file, return None
        print(f"Cannot find the file for subject {subject_id} with dominant hand = {is_dominant_hand} and labels = {which_labels}")
        return None

    # get the columns of the labels
    labels = get_labels_columes(actigraph_df)

    # for each labels, create a dataframe with one colume is the label and the other is the number of occurences
    summary = pd.DataFrame(columns=["label", "number_of_samples"])
    summary["label"] = labels
    summary["number_of_samples"] = [len(actigraph_df[actigraph_df[label] == 1]) for label in labels]
    
    # print out the summary
    # print("Summary for subject", subject_id, "with dominant hand = ", is_dominant_hand, "and labels = ", which_labels)
    # print(summary)
    total_sample = len(actigraph_df)
    # get the sum of the number of samples column
    total_number_of_labels = summary["number_of_samples"].sum()
    #write the total number of samples to the summary in a seperate row
    summary.loc[len(summary)] = ["total_samples", total_sample]
    #write the total number of labels to the summary in a seperate row
    summary.loc[len(summary)] = ["total_number_of_labels", total_number_of_labels]
    #save the summary to a csv file
    summary.to_csv(ROOT_DIR + f"data/PAAWS/labels_summary/{labels_type[which_labels]}/summary_{subject_id}_{is_dominant_hand}.csv", index=False)


# for label in range(1, 5):
#     for subject in range(20, 33):
#         try:
#             get_labels_summary(subject_id = subject, which_labels = label, is_dominant_hand = True)
#         except:
#             print("Error for subject", subject, "with dominant hand = True and labels = ", label)
