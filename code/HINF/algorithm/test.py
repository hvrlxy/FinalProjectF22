# impor the algorithm
from detection import DoubleThreshold
import os, sys
import pandas as pd

# set the parameters
soft_threshold = 50
hard_threshold = 500
wake_window = 10
motion_window = 3
wake_score_threshold = 0.5
motion_score_threshold = 0.5
activation_function = "linear"

# initialize the algorithm
algorithm = DoubleThreshold(soft_threshold, hard_threshold, wake_window, motion_window, wake_score_threshold, motion_score_threshold, activation_function)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../../.."
# for each subject
for subject in range(10, 33):
    print("Processing subject {}".format(subject))
    try:
        # create a folder for the subject at /data/PAAWS/HINF_results/plotDetectionByNight/
        if not os.path.exists(ROOT_DIR + f"/data/PAAWS/HINF_results/plotDetectionByNight/DS_{subject}"):
            os.makedirs(ROOT_DIR + f"/data/PAAWS/HINF_results/plotDetectionByNight/DS_{subject}")
        # for each hand (dominant and non-dominant), create a folder for the hand
        for hand in ["dominant_hand", "non_dominant_hand"]:
            if not os.path.exists(ROOT_DIR + f"/data/PAAWS/HINF_results/plotDetectionByNight/DS_{subject}/{hand}"):
                os.makedirs(ROOT_DIR + f"/data/PAAWS/HINF_results/plotDetectionByNight/DS_{subject}/{hand}")

            # get the data for each night
            # grab the list of files from the folder
            files = os.listdir(ROOT_DIR + f"/data/PAAWS/HINF_results/AUC_by_night/DS_{subject}/{hand}/")
            # for each file
            for file in files:
                if len(file.split("_")) < 3:
                    continue
                night = file.split("_")[2]
                # read the data
                auc_df = pd.read_csv(ROOT_DIR + f"/data/PAAWS/HINF_results/AUC_by_night/DS_{subject}/{hand}/{file}")
                # if the file is empty, skip
                if auc_df.shape[0] == 0:
                    continue
                # detect the wake and motion events
                auc_df = algorithm.detect(auc_df, plot = True, plot_path = ROOT_DIR + f"/data/PAAWS/HINF_results/plotDetectionByNight/DS_{subject}/{hand}/DS_{subject}_{hand}_{night}.png")
    except:
        print("Error processing subject {}".format(subject))
        continue