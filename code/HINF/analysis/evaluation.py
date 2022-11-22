import os, sys
import pandas as pd
import datetime as dt

#import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import warnings
warnings.filterwarnings("ignore")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

def get_confusion_matrix_for_subject(id: id, is_dominant_hand = True):
    if is_dominant_hand:
        results_path = ROOT_DIR + f'/data/PAAWS/HINF_results/results/DS_{id}/dominant_hand/wake_results.csv'
    else:
        results_path = ROOT_DIR + f'/data/PAAWS/HINF_results/results/DS_{id}/non_dominant_hand/wake_results.csv'
    results_df = pd.read_csv(results_path)

    # get the wake labels
    wake_label_path = ROOT_DIR + f'/data/PAAWS/HINF_results/wake_labels/DS_{id}.csv'
    df_wake_labels = pd.read_csv(wake_label_path)
    # get the unique dates in the wake labels
    dates = df_wake_labels['DATE'].unique()
    # filter out the night in the resuls_df and only take the one within the dates list
    results_df = results_df[results_df['date'].isin(dates)]
    results_dates = results_df['date'].unique()
    # filter out the DATE in the wake labels and only take the one within the results_dates list
    df_wake_labels = df_wake_labels[df_wake_labels['DATE'].isin(results_dates)]

    final_result_df = pd.DataFrame()
    # for each date in the results_dates
    for date in results_dates:
        # filter out the results_df and only take the one with the date
        df_results = results_df[results_df['date'] == date]
        # get the row in the wake labels with the date
        df_wake_label = df_wake_labels[df_wake_labels['DATE'] == date]
        # get the start time and stop time of the wake label
        start_time = df_wake_label['START_TIME_DATETIME'].values[0]
        stop_time = df_wake_label['STOP_TIME_DATETIME'].values[0]
        # convert the start time and stop time to epoch time
        start_time = dt.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S').timestamp()
        stop_time = dt.datetime.strptime(stop_time, '%Y-%m-%d %H:%M:%S').timestamp()
        # print(start_time, stop_time)
        #create a new column in the df_results called 'label_wake'
        df_results['label_wake'] = 1
        # for each row in the df_results
        for index, row in df_results.iterrows():
            # get the time of the row
            time = row['timestamp']
            # if the time is between the start time and the stop time
            if time >= start_time and time <= stop_time:
                # set the label_wake to 1
                df_results.at[index, 'label_wake'] = 0
        # append the df_results to the final_result_df
        final_result_df = final_result_df.append(df_results, ignore_index=True)
    # add datetime column
    final_result_df['datetime'] = final_result_df['timestamp'].apply(lambda x: dt.datetime.fromtimestamp(x))
    return final_result_df

def compute_vanilla_metrics_for_subjects(ids, is_dominant_hand = True):
    final_df = pd.DataFrame()
    for subject in ids:
        try:
            df = get_confusion_matrix_for_subject(subject, is_dominant_hand)
            final_df = final_df.append(df, ignore_index=True)
        except:
            continue
    y_true = final_df['label_wake']
    y_pred = final_df['is_wake']

    confusion_matrix_df = confusion_matrix(y_true, y_pred)
    accuracy_score_df = accuracy_score(y_true, y_pred)
    precision_score_df = precision_score(y_true, y_pred)
    recall_score_df = recall_score(y_true, y_pred)
    f1_score_df = f1_score(y_true, y_pred)

    false_positive = confusion_matrix_df[0][1]
    false_positive_rate = false_positive / (false_positive + confusion_matrix_df[0][0])
    return accuracy_score_df, precision_score_df, recall_score_df, f1_score_df, false_positive_rate

def compute_correct_prompting(id, is_dominant_hand = True, window_size = 5):
    if is_dominant_hand:
        results_path = ROOT_DIR + f'/data/PAAWS/HINF_results/results/DS_{id}/dominant_hand/wake_motion_events.csv'
    else:
        results_path = ROOT_DIR + f'/data/PAAWS/HINF_results/results/DS_{id}/non_dominant_hand/wake_motion_events.csv'
    results_df = pd.read_csv(results_path)

    # get the wake labels
    wake_label_path = ROOT_DIR + f'/data/PAAWS/HINF_results/wake_labels/DS_{id}.csv'
    df_wake_labels = pd.read_csv(wake_label_path)
    # get the unique dates in the wake labels
    dates = df_wake_labels['DATE'].unique()
    # filter out the night in the resuls_df and only take the one within the dates list
    results_df = results_df[results_df['night'].isin(dates)]
    results_dates = results_df['night'].unique()
    # filter out event wit htype = wake
    results_df = results_df[results_df['type'] == 'wake']
    # filter out the DATE in the wake labels and only take the one within the results_dates list
    df_wake_labels = df_wake_labels[df_wake_labels['DATE'].isin(results_dates)]

    total_days = len(results_dates)
    correct_prompt = 0
    window_seconds = window_size * 60
    # for each date in the results_dates
    for date in results_dates:
        # fulter out the results_df and only take the one with the date
        df_results = results_df[results_df['night'] == date]
        # get the stop_time of the sleep period
        df_wake_label = df_wake_labels[df_wake_labels['DATE'] == date]
        wake_time = df_wake_label['STOP_TIME_DATETIME'].values[0]
        #turn the wake time to datetime
        wake_time = dt.datetime.strptime(wake_time, '%Y-%m-%d %H:%M:%S')

        # for each of the wake event, see if any of them falls within 10 minutes of the wake time
        for index, row in df_results.iterrows():
            start_time = row['start']
            stop_time = row['end']
            # convert the start time and stop time to datetime from epoch time
            start_time = dt.datetime.fromtimestamp(start_time)
            stop_time = dt.datetime.fromtimestamp(stop_time)
            # check if start time is within 15 minutes of the wake time
            if abs(start_time - wake_time).total_seconds() <= window_seconds:
                #print start time, stop time, wake time
                # print("Scenario 1: ", start_time, stop_time, wake_time)
                correct_prompt += 1
                break
            # check if stop time is within 5 minutes of the wake time
            if abs(stop_time - wake_time).total_seconds() <= window_seconds:
                #print start time, stop time, wake time
                # print("Scenario 2: ", start_time, stop_time, wake_time)
                correct_prompt += 1
                break
            # check if wake time is within start time and stop time
            if start_time <= wake_time and stop_time >= wake_time:
                #print start time, stop time, wake time
                # print("Scenario 3: ", start_time, stop_time, wake_time)
                correct_prompt += 1
                break

    return correct_prompt, total_days

def compute_correct_prompting_rate(ids, is_dominant_hand = True, window_size = 5):
    total_instances = 0
    total_correct = 0
    for subject in ids:
        try:
            correct, total = compute_correct_prompting(subject, is_dominant_hand, window_size)
            total_instances += total
            total_correct += correct
        except:
            continue
    return total_correct / total_instances
# print("Compute metrics for subject 10 to 19, dominant hand")
# print(compute_vanilla_metrics_for_subjects(ids=[i for i in range(10, 20)], is_dominant_hand=True))

# print("Compute metrics for subject 10 to 19, non dominant hand")
# print(compute_vanilla_metrics_for_subjects(ids=[i for i in range(10, 20)], is_dominant_hand=False))

print("Print correct prompting rate for subject 10 to 19, dominant hand, window size 10 minutes")
print(compute_correct_prompting_rate(ids=[i for i in range(10, 20)], is_dominant_hand=True, window_size=10))

print("Print correct prompting rate for subject 10 to 19, non dominant hand, window size 10 minutes")
print(compute_correct_prompting_rate(ids=[i for i in range(10, 20)], is_dominant_hand=False, window_size=10))

print("Print correct prompting rate for subject 10 to 19, dominant hand, window size 3 minutes")
print(compute_correct_prompting_rate(ids=[i for i in range(10, 20)], is_dominant_hand=True, window_size=3))

print("Print correct prompting rate for subject 10 to 19, non dominant hand, window size 3 minutes")
print(compute_correct_prompting_rate(ids=[i for i in range(10, 20)], is_dominant_hand=False, window_size=3))

print("Print correct prompting rate for subject 10 to 19, dominant hand, window size 1 minutes")
print(compute_correct_prompting_rate(ids=[i for i in range(10, 20)], is_dominant_hand=True, window_size=1))

print("Print correct prompting rate for subject 10 to 19, non dominant hand, window size 1 minutes")
print(compute_correct_prompting_rate(ids=[i for i in range(10, 20)], is_dominant_hand=False, window_size=1))