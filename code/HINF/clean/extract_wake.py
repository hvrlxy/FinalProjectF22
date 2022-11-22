import os, sys
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../../../"

def processing_posture_label(subject_id):
    label_file = ROOT_DIR + "data/PAAWS/raw_labels/DS_" + str(subject_id) + "/combined_POSTURE_corr.csv"
    df = pd.read_csv(label_file)
    #  only grab the START_TIME and END_TIME AND THE PREDICTED LABEL
    df = df[['START_TIME', 'STOP_TIME', 'PREDICTION']]
    # convert the start and stop time to datetime
    df['START_TIME'] = pd.to_datetime(df['START_TIME'])
    df['STOP_TIME'] = pd.to_datetime(df['STOP_TIME'])
    #convert them to epoch time in seconds  
    df['START_TIME'] = df['START_TIME'].astype(int) // 10**9
    df['STOP_TIME'] = df['STOP_TIME'].astype(int) // 10**9

    #initialise the new dataframe called sleep_df
    sleep_df = pd.DataFrame(columns=['START_TIME', 'STOP_TIME', 'DATE'])
    # for each row, compare it to the previous row. If the start time is 
    # 4 hours after the previous row's stop time, then it is considered a sleep period
    for index, row in df.iterrows():
        if index == 0:
            continue
        if (row['START_TIME'] - df.iloc[index-1]['STOP_TIME']) > 14400:
            # get the date of the sleep period
            date = df.iloc[index]['START_TIME']
            # convert to datetime
            date = pd.to_datetime(date, unit='s')
            # get the date only
            date = date.date()
            # append the sleep period to the sleep_df
            sleep_df = sleep_df.append({'START_TIME': df.iloc[index-1]['STOP_TIME'], 'STOP_TIME': row['START_TIME'], 'DATE': date}, ignore_index=True)
        # add another columes with the start_time converted to datetime
        sleep_df['START_TIME_DATETIME'] = pd.to_datetime(sleep_df['START_TIME'], unit='s')
        # add another columes with the stop_time converted to datetime
        sleep_df['STOP_TIME_DATETIME'] = pd.to_datetime(sleep_df['STOP_TIME'], unit='s')
        # get the hour of the stop time
        sleep_df['STOP_TIME_HOUR'] = sleep_df['STOP_TIME_DATETIME'].dt.hour
        # remove any data point with hour more than 13
        sleep_df = sleep_df[sleep_df['STOP_TIME_HOUR'] < 13]
        #drop the hour column
        sleep_df = sleep_df.drop(columns=['STOP_TIME_HOUR'])
    # save the sleep_df to a csv file
    sleep_df.to_csv(ROOT_DIR + f"data/PAAWS/HINF_results/wake_labels/DS_{str(subject_id)}.csv", index=False)

for i in range(20, 33):
    try:
        processing_posture_label(i)
    except:
        print(f"Error with subject {i}")