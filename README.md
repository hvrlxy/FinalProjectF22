# FinalProjectF22

This repository contains the code and datasets I used for both of my final project fall 2022 at Northeastern University. There are two dataset being used: the NHANES datasets which consists of 16 000 people'swrist-worn actigraphy data. The PAAWS dataset is an on-going research project from the mHealth lab at Northeastern University.

## Data
All the data is contains inside the [data](/data/) folder. There are two folders: the [PAAWS](/data/PAAWS) folder contains the PAAWS raw and filtered dataset, along with the labels. the [NHANES](/data/NHANES) folder contains the raw and cleaned actigraphy data.

Inside the PAAWS folder, there are a couples of subfolders, with the follwing content:

- [RAW](/data/PAAWS/raw/) folder : contains the raw data from participant 10 to 32. The PAAWS dataset contains multisensor data, along with sleep data and questionaires. However, for the sake of the project, I will only be using wrist-worn acigraphy data. Inside the folder are subfolders for each participants, with the syntax <DS_{subject_id}>. There should be 2 csv files inside: one for the dominant/left hand and one for the non-dominant/right hand.
- [FILTELRED](/data/PAAWS/filtered/) folder : contains the raw actigraphy data with timestamp included. Each participants should have 2 csv files for dominant and non-dominant hand.
- [LABELS](/data/PAAWS/labels/) folder : contains the labels for the dataset. There are labels for nearly 30 participants, but the one that we are interested in is participants 10 to 32. For those participants, the label folder should include labels for individual days (Day1 to Day8), and the combined, summary labels csv files.
- [FILTERED_LABELS](/data/PAAWS/filtered_labels/) folder : contains the csv files for the clean labels. Inside this folder there are 4 subfolders: [BehavioralPattern](/data/PAAWS/filtered_labels/BehavioralPattern),[HighLevelBehavioralPattern](/data/PAAWS/filtered_labels/HighLevelBehavioralPattern), [PhysicalActivity](/data/PAAWS/filtered_labels/PhysicalActivity) and [Posture](/data/PAAWS/filtered_labels/Posture). Inside each of the 4 subfolders contains csv files for the labels of all participants. The csv files should follows the format [<START_TIME>, <STOP_TIME>, <LABEL_1>, <LABEL_2>, ...]
- [ACTIGRAPHY_FEATURES](/data/PAAWS/actigraphy_features/) folder : contains the cleaned actigraphy data after filtering, segmentation and feature generation. There should be roughly 60k lines in each of the csv file.
- [LABELED_ACTIGRAPHY](/data/PAAWS/labeled_actigraphy/) folder : contains the cleaned actigraphy data with the labels synced. Each participant should have 2 files, one for dominant/left hand and the other for non-dominant/right hand.