import os, sys
import subprocess
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# get the directory of this file
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../"

def processing_behavior_param_labels(subject_id):
    '''
    This function processes the behavioral parameter labels for a given subject
    Parameters:
        subject_id: the id of the subject
    Returns:
        None
    '''
    label_file = ROOT_DIR + "data/PAAWS/raw_labels/DS_" + str(subject_id) + "/combined_Behavioral Parameters_corr.csv"
    df = pd.read_csv(label_file)
    
    #  only grab the START_TIME and END_TIME AND THE PREDICTED LABEL
    df = df[['START_TIME', 'STOP_TIME', 'PREDICTION']]
    # seperate the labels in the prediction column by /
    df['PREDICTION'] = df['PREDICTION'].str.split('/')
    # get all unique labels in the prediction column
    unique_labels = df['PREDICTION'].explode().unique()
    # create a new column for each unique label
    for label in unique_labels:
        df[label] = df['PREDICTION'].apply(lambda x: 1 if label in x else 0)
    # drop the prediction column
    df = df.drop(columns=['PREDICTION'])
    # save the new dataframe to a csv file
    df.to_csv(ROOT_DIR + f"data/PAAWS/filtered_labels/BehavioralPattern/DS_{subject_id}_combined_BehavioralParameters_corr.csv", index=False)
    # print out the first 5 rows
    # print(df.head())

def processing_high_level_behavior_label(subject_id):
    label_file = ROOT_DIR + "data/PAAWS/raw_labels/DS_" + str(subject_id) + "/combined_HIGH LEVEL BEHAVIOR_corr.csv"
    df = pd.read_csv(label_file)
    #  only grab the START_TIME and END_TIME AND THE PREDICTED LABEL
    df = df[['START_TIME', 'STOP_TIME', 'PREDICTION']]
    # seperate the labels in the prediction column by /
    df['PREDICTION'] = df['PREDICTION'].str.split('/')
    # get all unique labels in the prediction column
    unique_labels = df['PREDICTION'].explode().unique()
    # create a new column for each unique label
    for label in unique_labels:
        df[label] = df['PREDICTION'].apply(lambda x: 1 if label in x else 0)
    # drop the prediction column
    df = df.drop(columns=['PREDICTION'])
    # save the new dataframe to a csv file
    df.to_csv(ROOT_DIR + f"data/PAAWS/filtered_labels/HighLevelBehavioralPattern/DS_{subject_id}_combined_HighLevelBehavioralPatterns.csv", index=False)
    # print out the first 5 rows
    # print(df.head(5))

def processing_physical_activity_label(subject_id):
    label_file = ROOT_DIR + "data/PAAWS/raw_labels/DS_" + str(subject_id) + "/combined_PA TYPE_corr.csv"
    df = pd.read_csv(label_file)
    #  only grab the START_TIME and END_TIME AND THE PREDICTED LABEL
    df = df[['START_TIME', 'STOP_TIME', 'PREDICTION']]
    # seperate the labels in the prediction column by /
    df['PREDICTION'] = df['PREDICTION'].str.split('/')
    # get all unique labels in the prediction column
    unique_labels = df['PREDICTION'].explode().unique()
    # create a new column for each unique label
    for label in unique_labels:
        df[label] = df['PREDICTION'].apply(lambda x: 1 if label in x else 0)
    # drop the prediction column
    df = df.drop(columns=['PREDICTION'])
    # save the new dataframe to a csv file
    df.to_csv(ROOT_DIR + f"data/PAAWS/filtered_labels/PhysicalActivity/DS_{subject_id}_combined_PhysicalActivity.csv", index=False)
    # print out the first 5 rows
    # print(df.head(5))

def processing_posture_label(subject_id):
    label_file = ROOT_DIR + "data/PAAWS/raw_labels/DS_" + str(subject_id) + "/combined_POSTURE_corr.csv"
    df = pd.read_csv(label_file)
    #  only grab the START_TIME and END_TIME AND THE PREDICTED LABEL
    df = df[['START_TIME', 'STOP_TIME', 'PREDICTION']]
    # seperate the labels in the prediction column by /
    df['PREDICTION'] = df['PREDICTION'].str.split('/')
    # get all unique labels in the prediction column
    unique_labels = df['PREDICTION'].explode().unique()
    # create a new column for each unique label
    for label in unique_labels:
        df[label] = df['PREDICTION'].apply(lambda x: 1 if label in x else 0)
    # drop the prediction column
    df = df.drop(columns=['PREDICTION'])
    # save the new dataframe to a csv file
    df.to_csv(ROOT_DIR + f"data/PAAWS/filtered_labels/Posture/DS_{subject_id}_combined_Posture.csv", index=False)
    # print out the first 5 rows
    # print(df.head(5))

def generate_labels_datasets():
    # generate the labels for each subject from 9 to 32
    for subject_id in range(9, 33):
        try:
            processing_behavior_param_labels(subject_id)
            #print some success message
            print(f"Successfully processed the behavioral parameter labels for subject {subject_id}")
        except:
            print(f"Error processing behavioral parameters for subject {subject_id}")
        
        try:
            processing_high_level_behavior_label(subject_id)
            #print some success message
            print(f"Successfully processed the high level behavioral labels for subject {subject_id}")
        except:
            print(f"Error processing high level behavior for subject {subject_id}")

        try:
            processing_physical_activity_label(subject_id)
            #print some success message
            print(f"Successfully processed the physical activity labels for subject {subject_id}")
        except:
            print(f"Error processing physical activity for subject {subject_id}")
        
        try:
            processing_posture_label(subject_id)
            #print some success message
            print(f"Successfully processed the posture labels for subject {subject_id}")
        except:
            print(f"Error processing posture for subject {subject_id}")

