import os, sys
import scipy
from scipy.signal import find_peaks
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import datetime
import glob
import warnings

warnings.filterwarnings('ignore')

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../../"

class ActigraphSummary:
    def __init__(self, root_path):
        self.path = root_path 

    def get_file_name(self, subject_id: int):
        '''
        This function takes in the subject's id and return the list of timestamp-synced actigraph files for the subject
        :params: subject_id: id of the subject
        '''
        folder_path =  self.path + "data/PAAWS/filtered/"
        files_lst = os.listdir(folder_path)
        #find if any files named DS_subject_id is in the folder
        subject_files = [file for file in files_lst if f'DS_{subject_id}' in file]
        print(f"Files found for subject {subject_id}: ", subject_files)
        return subject_files

    def print_head_file(self, subject_id):
        '''
        This function takes in the subject's id and print the head of the actigraph file
        :params: subject_id: id of the subject
        :return: None
        '''
        subject_files = self.get_file_name(subject_id=subject_id)
        for file in subject_files:
            print("Printing out the first 5 lines for file: ", file)
            file_df = pd.read_csv(f"{self.path}/data/PAAWS/filtered/{file}")
            # rename the axis to be timestamp, x, y, z
            # file_df.rename
            # print(file_df.head(5))
            #print rows 100 to 110
            print(file_df.iloc[80000:80010])
            # print the first 5 timestamp
            print(list(file_df['timestamp'].head(5)))

    def apply_filter_to_all_axis(self, subject_id, actigraphy_df):
        '''
        This function takes in the subject's id and apply the low pass filter to all the axis. This function return the filtered data.
        :params: subject_id: id of the subject
        :params: actigraphy_df: the actigraphy dataframe
        :return: filtered data
        '''
        # rename the axis to be timestamp, x, y, z
        actigraphy_df.rename(columns={'timestamp': 'timestamp', 'Accelerometer X': 'x', 'Accelerometer Y': 'y', 'Accelerometer Z': 'z'}, inplace=True)
        # apply low pass filter to all the axis
        actigraphy_df['x'] = self.moving_average_filtered(actigraphy_df['x'])
        actigraphy_df['y'] = self.moving_average_filtered(actigraphy_df['y'])
        actigraphy_df['z'] = self.moving_average_filtered(actigraphy_df['z'])

        #save the filtered data to a csv file
        # actigraphy_df.to_csv(f"{self.path}/data/PAAWS/filtered/low_pass_filtered/filtered_{subject_id}.csv", index=False)
        return actigraphy_df

    def segment_and_add_features(self, subject_id, is_dominant_hand=True):
        '''
        This function takes in the subject's id, segment the data into 10s window and add features to the data.
        Return a new dataframe with the features added and the timestamp of the first sample in the segment
        :params: subject_id: id of the subject
        :params: is_dominant_hand: whether the data we are looking at is dominant/right hand or not
        :return: a new dataframe with the features added and the timestamp of the first sample in the segment
        '''
        subject_files = self.get_file_name(subject_id=subject_id)
        file = subject_files[0]
        if not is_dominant_hand:
            file = subject_files[1]
        print(f"Applying low pass filter to actigraph data of subject {subject_id} with dominant hand = {is_dominant_hand}: ")
        actigraphy_df = pd.read_csv(f"{self.path}/data/PAAWS/filtered/{file}")
        # rename the axis to be timestamp, x, y, z
        actigraphy_df.rename(columns={'timestamp': 'timestamp', 'Accelerometer X': 'x', 'Accelerometer Y': 'y', 'Accelerometer Z': 'z'}, inplace=True)
        actigraphy_df = self.apply_filter_to_all_axis(subject_id, actigraphy_df)
        # loop through the data in 10s window (80Hz * 10s = 800 samples)
        window_size = 800
        length_data = len(actigraphy_df)

        # create a new dataframe to store the features
        new_df = pd.DataFrame()
        timestamp = []
        for i in range(0, length_data, window_size):
            timestamp.append(actigraphy_df['timestamp'][i])
        # print the first and last timestamp of the actigraphy data
        print(f"First timestamp: {actigraphy_df['timestamp'][0]}")
        print(f"Last timestamp: {actigraphy_df['timestamp'][length_data-1]}")
        print("Length of the actigraphy data: ", length_data)
        new_df['timestamp'] = timestamp
        # gểnate features for x axis
        x_series = actigraphy_df['x']
        # loggin the mean of the x axis
        print("Calculating the mean of the x axis")
        new_df['x_mean'] = x_series.groupby(np.arange(len(x_series.index)) // 800).mean()
        # print(new_df['x_mean'])
        # loggin the std of the x axis
        print("Calculating the std of the x axis")
        new_df['x_std'] = x_series.groupby(np.arange(len(x_series.index)) // 800).std()
        # loggin the min of the x axis
        print("Calculating the min of the x axis")
        new_df['x_min'] = x_series.groupby(np.arange(len(x_series.index)) // 800).min()
        # loggin the max of the x axis
        print("Calculating the max of the x axis")
        new_df['x_max'] = x_series.groupby(np.arange(len(x_series.index)) // 800).max()
        # loggin the median of the x axis
        print("Calculating the median of the x axis")
        new_df['x_median'] = x_series.groupby(np.arange(len(x_series.index)) // 800).median()
        # loggin the skewness of the x axis
        print("Calculating the skewness of the x axis")
        new_df['x_skew'] = x_series.groupby(np.arange(len(x_series.index)) // 800).skew()
        # get the frequency domain features of the x axis
        fft_features_x = x_series.groupby(np.arange(len(x_series.index)) // 800).apply(self.get_freq_domain_features)
        fft_x_cols = ['x_fft_dc', 'x_fft_mean', 'x_fft_std', 'x_fft_aad', 'x_fft_min', 'x_fft_max', 'x_fft_maxmin_diff', 'x_fft_median', 
                      'x_fft_mad', 'x_fft_IQR', 'x_fft_neg_count', 'x_fft_pos_count', 'x_fft_above_mean', 'x_fft_num_peaks', 'x_fft_skew', 
                      'x_fft_kurtosis', 'x_fft_energy', 'x_fft_sma']
        for i in range(len(fft_x_cols)):
            print(f"Calculating the {fft_x_cols[i]} of the x axis")
            new_df[fft_x_cols[i]] = fft_features_x.apply(lambda x: x[i])
        # get the time domain features of the y axis
        y_series = actigraphy_df['y']
        print("Calculating the mean of the y axis")
        new_df['y_mean'] = actigraphy_df['y'].groupby(np.arange(len(actigraphy_df.index)) // 800).mean()
        print("Calculating the std of the y axis")
        new_df['y_std'] = y_series.groupby(np.arange(len(y_series.index)) // 800).std()
        # loggin the min of the y axis
        print("Calculating the min of the y axis")
        new_df['y_min'] = y_series.groupby(np.arange(len(y_series.index)) // 800).min()
        # loggin the max of the y axis
        print("Calculating the max of the y axis")
        new_df['y_max'] = y_series.groupby(np.arange(len(y_series.index)) // 800).max()
        # loggin the median of the y axis
        print("Calculating the median of the y axis")
        new_df['y_median'] = y_series.groupby(np.arange(len(y_series.index)) // 800).median()
        # loggin the skewness of the y axis
        print("Calculating the skewness of the y axis")
        new_df['y_skew'] = y_series.groupby(np.arange(len(y_series.index)) // 800).skew()
        # get the frequency domain features of the y axis
        fft_features_x = y_series.groupby(np.arange(len(y_series.index)) // 800).apply(self.get_freq_domain_features)
        fft_y_cols = ['y_fft_dc', 'y_fft_mean', 'y_fft_std', 'y_fft_aad', 'y_fft_min', 'y_fft_max', 'y_fft_maxmin_diff', 'y_fft_median', 
                      'y_fft_mad', 'y_fft_IQR', 'y_fft_neg_count', 'y_fft_pos_count', 'y_fft_above_mean', 'y_fft_num_peaks', 'y_fft_skew', 
                      'y_fft_kurtosis', 'y_fft_energy', 'y_fft_sma']
        for i in range(len(fft_y_cols)):
            print(f"Calculating the {fft_y_cols[i]} of the y axis")
            new_df[fft_y_cols[i]] = fft_features_x.apply(lambda x: x[i])
        # get the time domain features of the z axis
        z_series = actigraphy_df['z']
        print("Calculating the mean of the z axis")
        new_df['z_mean'] = z_series.groupby(np.arange(len(actigraphy_df.index)) // 800).mean()
        print("Calculating the std of the z axis")
        new_df['z_std'] = z_series.groupby(np.arange(len(z_series.index)) // 800).std()
        # loggin the min of the z axis
        print("Calculating the min of the z axis")
        new_df['z_min'] = z_series.groupby(np.arange(len(z_series.index)) // 800).min()
        # loggin the max of the z axis
        print("Calculating the max of the z axis")
        new_df['z_max'] = z_series.groupby(np.arange(len(z_series.index)) // 800).max()
        # loggin the median of the z axis
        print("Calculating the median of the z axis")
        new_df['z_median'] = z_series.groupby(np.arange(len(z_series.index)) // 800).median()
        # loggin the skewness of the z axis
        print("Calculating the skewness of the z axis")
        new_df['z_skew'] = z_series.groupby(np.arange(len(z_series.index)) // 800).skew()
        # get the frequency domain features of the z axis
        fft_features_x = z_series.groupby(np.arange(len(z_series.index)) // 800).apply(self.get_freq_domain_features)
        fft_z_cols = ['z_fft_dc', 'z_fft_mean', 'z_fft_std', 'z_fft_aad', 'z_fft_min', 'z_fft_max', 'z_fft_maxmin_diff', 'z_fft_median', 
                      'z_fft_mad', 'z_fft_IQR', 'z_fft_neg_count', 'z_fft_pos_count', 'z_fft_above_mean', 'z_fft_num_peaks', 'z_fft_skew', 
                      'z_fft_kurtosis', 'z_fft_energy', 'z_fft_sma']
        for i in range(len(fft_z_cols)):
            print(f"Calculating the {fft_z_cols[i]} of the z axis")
            new_df[fft_z_cols[i]] = fft_features_x.apply(lambda x: x[i])        
        
        # new_df['z_fft_kurtosis'] = z_series.groupby(np.arange(len(z_series.index)) // 800).apply(self.get_freq_domain_features).apply(pd.Series).kurtosis(axis=1)
        #  save the new dataframe to csv
        if is_dominant_hand:
            new_df.to_csv(f'{self.path}/data/PAAWS/actigraphy_features/DS_{subject_id}_actigraphy_features_dominant.csv', index=False)
        else:
            new_df.to_csv(f'{self.path}/data/PAAWS/actigraphy_features/DS_{subject_id}_actigraphy_features_non_dominant.csv', index=False)
        return new_df

    def low_pass_filter(self, data, cutoff = 3, fs = 20, order=5):
        '''
        This function takes in the data, cutoff frequency, sampling frequency and order of the filter
        and returns the filtered data. The actigaphy data is 80Hz
        :params: data: the data to be filtered
        :params: cutoff: the cutoff frequency
        :params: fs: the sampling frequency
        :params: order: the order of the filter
        :return: filtered data
        '''
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        # Get the filter coefficients 
        b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
        y = scipy.signal.lfilter(b, a, data)
        return y

    def moving_average_filtered(self, data, window_size = 20):
        '''
        This function takes in the data and the window size and returns the moving average filtered data
        :params: data: the data to be filtered
        :params: window_size: the window size
        :return: filtered data
        '''
        filtered_value = np.convolve(data, np.ones((window_size,))/window_size, mode='valid')
        # add the window_sixe - 1 values to the beginning of the filtered data
        filtered_value = np.concatenate((np.full((window_size - 1), np.nan), filtered_value))
        return filtered_value

    def get_time_series_features(self, signal):
        window_size = len(signal)
        # mean
        sig_mean = np.mean(signal)
        # standard deviation
        sig_std = np.std(signal)
        # avg absolute difference
        sig_aad = np.mean(np.absolute(signal - np.mean(signal)))
        # min
        sig_min = np.min(signal)
        # max
        sig_max = np.max(signal)
        # max-min difference
        sig_maxmin_diff = sig_max - sig_min
        # median
        sig_median = np.median(signal)
        # median absolute deviation
        sig_mad = np.median(np.absolute(signal - np.median(signal)))
        # Inter-quartile range
        sig_IQR = np.percentile(signal, 75) - np.percentile(signal, 25)
        # negative count
        sig_neg_count = np.sum(s < 0 for s in signal)
        # positive count
        sig_pos_count = np.sum(s > 0 for s in signal)
        # values above mean
        sig_above_mean = np.sum(s > sig_mean for s in signal)
        # number of peaks
        sig_num_peaks = len(find_peaks(signal)[0])
        # skewness
        sig_skew = stats.skew(signal)
        # kurtosis
        sig_kurtosis = stats.kurtosis(signal)
        # energy
        sig_energy = np.sum(s ** 2 for s in signal) / window_size
        # signal area
        sig_sma = np.sum(signal) / window_size

        return [sig_mean, sig_std, sig_aad, sig_min, sig_max, sig_maxmin_diff, sig_median, sig_mad, sig_IQR, sig_neg_count, sig_pos_count, sig_above_mean, sig_num_peaks, sig_skew, sig_kurtosis, sig_energy, sig_sma]

    def get_freq_domain_features(self, signal):
        all_fft_features = []
        window_size = len(signal)
        signal_fft = np.abs(np.fft.fft(signal))
        # Signal DC component
        sig_fft_dc = signal_fft[0]
        # aggregations over the fft signal
        fft_feats = self.get_time_series_features(signal_fft[1:int(window_size / 2) + 1])

        all_fft_features.append(sig_fft_dc)
        all_fft_features.extend(fft_feats)
        return all_fft_features

    def test_filter(self, subject_id):
        '''
        This function takes in the subject_id, extract the first 10 minutes of data (80Hz) and plot the original and filtered data
        :params: subject_id: the id of the subject
        '''
        subject_file = self.get_file_name(subject_id=subject_id)[0]
        file_df = pd.read_csv(f"{self.path}/data/PAAWS/filtered/{subject_file}")
        # print out the columns
        # print(file_df.columns)
        # rename the axis to be timestamp, x, y, z
        file_df.rename(columns={'timestamp': 'timestamp', 'Accelerometer X': 'x', 'Accelerometer Y': 'y', 'Accelerometer Z': 'z'}, inplace=True)
        # get the first 30 seconds of data
        x_axis = file_df['x']
        x_filtered = self.moving_average_filtered(x_axis)
        print(x_axis[10000000:10000000 +30*80])
        # plot the original data for x axis in one subplot
        plt.subplot(2,1,1)
        plt.plot(x_axis[10000000:10000000 +30*80])
        plt.title("Original data")
        # plot the filtered data for x axis in another subplot
        plt.subplot(2,1,2)
        print(x_filtered[10000000:10000000 +30*80])
        plt.plot(x_filtered[10000000:10000000 +30*80])
        plt.title("Filtered data")
        plt.show()


accelSummary = ActigraphSummary(ROOT_DIR)
# accelSummary.segment_and_add_features(10)
for i in range(10, 11):
    accelSummary.segment_and_add_features(i)