import os
import sys
import warnings
import time
import pandas as pd

warnings.filterwarnings("ignore")

RAW_DIR = "/Users/hale/Desktop/FinalProjectF22-1/data/NHANES/raw/raw_sensor_file"
FEATURES_DIR = "/Users/hale/Desktop/FinalProjectF22-1/data/NHANES/features_extracted"

def get_raw_files():
    raw_files = []
    ids = []
    for file in os.listdir(RAW_DIR):
        if ".csv" in file:
            raw_files.append(file)
            ids.append(file.split(".")[0])
    return raw_files, ids

def get_features_files():
    features_files = []
    ids = []
    for file in os.listdir(FEATURES_DIR):
        if ".csv" in file:
            features_files.append(file)
            ids.append(file.split(".")[0])
    return features_files, ids

def remove_processed_files():
    raw_files, raw_ids = get_raw_files()
    features_files, features_ids = get_features_files()
    for i in range(len(raw_ids)):
        if raw_ids[i] in features_ids:
            os.remove(RAW_DIR + "/" + raw_files[i])

remove_processed_files()
raw_files, raw_ids = get_raw_files()
while (True):
    if (len(raw_files) > 0):
        # shuffle the files list
        raw_ids = pd.Series(raw_ids).sample(frac=1).values
        # get the first file
        file = raw_ids[0]
        # extract the features
        os.system("python3 /Users/hale/Desktop/FinalProjectF22-1/code/NHANES/clean/process_raw_data.py " + file)
        # remove the processed file
        remove_processed_files()
        # get the new list of files
        raw_files, raw_ids = get_raw_files()
    # sleep for 1 minute
    time.sleep(60)