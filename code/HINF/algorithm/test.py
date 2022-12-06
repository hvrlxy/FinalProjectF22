# impor the algorithm
from detection import DoubleThreshold
import os, sys
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
# set the parameters
soft_threshold = 50
hard_threshold = 500
wake_window = 10
motion_window = 3
wake_score_threshold = 0.5
motion_score_threshold = 0.5
activation_function = "linear"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../../.."

def run_algorithm(soft_threshold, hard_threshold, wake_window, motion_window, wake_score_threshold, motion_score_threshold, activation_function):
    algorithm = DoubleThreshold(soft_threshold, hard_threshold, wake_window, motion_window, wake_score_threshold, motion_score_threshold, activation_function)

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
                    night = night.replace(".csv", "")
                    # read the data
                    auc_df = pd.read_csv(ROOT_DIR + f"/data/PAAWS/HINF_results/AUC_by_night/DS_{subject}/{hand}/{file}")
                    # if the file is empty, skip
                    if auc_df.shape[0] == 0:
                        continue
                    # detect the wake and motion events
                    wake_events, motion_events, wake_results_df, motion_results_df = algorithm.detect(auc_df, plot = False, plot_path = ROOT_DIR + f"/data/PAAWS/HINF_results/plotDetectionByNight/DS_{subject}/{hand}/DS_{subject}_{hand}_{night}.png")
                    # save the wake and motion events
                    # search if the subject folder exists in the results folder
                    if not os.path.exists(ROOT_DIR + f"/data/PAAWS/HINF_results/results/DS_{subject}"):
                        os.makedirs(ROOT_DIR + f"/data/PAAWS/HINF_results/results/DS_{subject}")
                    # search if the hand folder exists in the results folder
                    if not os.path.exists(ROOT_DIR + f"/data/PAAWS/HINF_results/results/DS_{subject}/{hand}"):
                        os.makedirs(ROOT_DIR + f"/data/PAAWS/HINF_results/results/DS_{subject}/{hand}")
                        # see if the file exists
                    if not os.path.exists(ROOT_DIR + f"/data/PAAWS/HINF_results/results/DS_{subject}/{hand}/wake_motion_events.csv"):
                        # create the file
                        wake_motion_events = pd.DataFrame(columns = ["start", "end", "type", "night"])
                        # add the events
                        for wake_event in wake_events:
                            wake_motion_events = wake_motion_events.append({"start": wake_event[0], "end": wake_event[1], "type": "wake", "night": night}, ignore_index = True)
                        for motion_event in motion_events:
                            wake_motion_events = wake_motion_events.append({"start": motion_event[0], "end": motion_event[1], "type": "motion", "night": night}, ignore_index = True)

                        wake_motion_events.to_csv(ROOT_DIR + f"/data/PAAWS/HINF_results/results/DS_{subject}/{hand}/wake_motion_events.csv", index = False)
                    else:
                        # append to the file
                        wake_motion_events = pd.read_csv(ROOT_DIR + f"/data/PAAWS/HINF_results/results/DS_{subject}/{hand}/wake_motion_events.csv")
                        for wake_event in wake_events:
                            wake_motion_events = wake_motion_events.append({"start": wake_event[0], "end": wake_event[1], "type": "wake", "night": night}, ignore_index = True)
                        for motion_event in motion_events:
                            wake_motion_events = wake_motion_events.append({"start": motion_event[0], "end": motion_event[1], "type": "motion", "night": night}, ignore_index = True)
                        
                        wake_motion_events.to_csv(ROOT_DIR + f"/data/PAAWS/HINF_results/results/DS_{subject}/{hand}/wake_motion_events.csv", index = False)

                    # search if the csv file for wake_results exists
                    if not os.path.exists(ROOT_DIR + f"/data/PAAWS/HINF_results/results/DS_{subject}/{hand}/wake_results.csv"):
                        # create the file
                        wake_results_df.to_csv(ROOT_DIR + f"/data/PAAWS/HINF_results/results/DS_{subject}/{hand}/wake_results.csv", index = False)
                    else:
                        # append to the file
                        wake_results = pd.read_csv(ROOT_DIR + f"/data/PAAWS/HINF_results/results/DS_{subject}/{hand}/wake_results.csv")
                        wake_results = wake_results.append(wake_results_df, ignore_index = True)
                        wake_results.to_csv(ROOT_DIR + f"/data/PAAWS/HINF_results/results/DS_{subject}/{hand}/wake_results.csv", index = False)

                    # search if the csv file for motion_results exists
                    if not os.path.exists(ROOT_DIR + f"/data/PAAWS/HINF_results/results/DS_{subject}/{hand}/motion_results.csv"):
                        # create the file
                        motion_results_df.to_csv(ROOT_DIR + f"/data/PAAWS/HINF_results/results/DS_{subject}/{hand}/motion_results.csv", index = False)
                    else:
                        # append to the file
                        motion_results = pd.read_csv(ROOT_DIR + f"/data/PAAWS/HINF_results/results/DS_{subject}/{hand}/motion_results.csv")
                        motion_results = motion_results.append(motion_results_df, ignore_index = True)
                        motion_results.to_csv(ROOT_DIR + f"/data/PAAWS/HINF_results/results/DS_{subject}/{hand}/motion_results.csv", index = False)
        except Exception as e:
            print(f"RUNNING ALGORITHM: Error processing subject {subject}".format(subject))
            print(e)
            continue

# if __name__ == "__main__":
#     # get the arguments
#     if len(sys.argv) < 8:
#         print("Usage: python3 detectEventsByNight.py <soft_threshold> <hard_threshold> <wake_window> <motion_window> <wake_score_threshold> <motion_score_threshold> <activation_function>")
#         exit(1)

#     soft_threshold = float(sys.argv[1])
#     hard_threshold = float(sys.argv[2])
#     wake_window = int(sys.argv[3])
#     motion_window = int(sys.argv[4])
#     wake_score_threshold = float(sys.argv[5])
#     motion_score_threshold = float(sys.argv[6])
#     activation_function = sys.argv[7]

#     # run the main function
#     run_algorithm(soft_threshold, hard_threshold, wake_window, motion_window, wake_score_threshold, motion_score_threshold, activation_function)


run_algorithm(50, 500, 10, 3, 0.5, 0.5, 'linear')

