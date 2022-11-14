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

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../"

class ActigraphSummary:
    def __init__(self, root_path):
        self.path = root_path 

    def get_file_name(self, subject_id: int):
        '''
        This function takes in the subject's id and return the list of timestamp-synced actigraph files for the subject
        ::params:subject_id: id of the subject
        '''
        folder_path =  self.path + "data/filtered/"
        files_lst = os.listdir(folder_path)
        #find if any files named DS_subject_id is in the folder
        subject_files = [file for file in files_lst if f'DS_{subject_id}' in file]
        print(f"Files found for subject {subject_id}: ", subject_files)
        return subject_files

    def print_head_file(self, subject_id):
        subject_files = self.get_file_name(subject_id=subject_id)
        for file in subject_files:
            print("Printing out the first 5 lines for file: ", file)
            file_df = pd.read_csv(f"{self.path}/data/filtered/{file}")
            # rename the axis to be timestamp, x, y, z
            # file_df.rename
            # print(file_df.head(5))
            #print rows 100 to 110
            print(file_df.iloc[80000:80010])
            # print the first 5 timestamp
            print(list(file_df['timestamp'].head(5)))

    def apply_filter_to_all_axis(self, actigraphy_df):
        '''
        This function takes in the subject's id and apply the low pass filter to all the axis. This function return the filtered data.
        '''
        # rename the axis to be timestamp, x, y, z
        actigraphy_df.rename(columns={'timestamp': 'timestamp', 'Accelerometer X': 'x', 'Accelerometer Y': 'y', 'Accelerometer Z': 'z'}, inplace=True)
        # apply low pass filter to all the axis
        actigraphy_df['x'] = self.low_pass_filter(actigraphy_df['x'])
        actigraphy_df['y'] = self.low_pass_filter(actigraphy_df['y'])
        actigraphy_df['z'] = self.low_pass_filter(actigraphy_df['z'])
        return actigraphy_df

    def segment_and_add_features(self, subject_id, is_dominant_hand=True):
        '''
        This function takes in the subject's id, segment the data into 10s window and add features to the data.
        Return a new dataframe with the features added and the timestamp of the first sample in the segment
        '''
        subject_files = self.get_file_name(subject_id=subject_id)
        file = subject_files[0]
        if not is_dominant_hand:
            file = subject_files[1]
        print(f"Applying low pass filter to actigraph data of subject {subject_id} with dominant hand = {is_dominant_hand}: ")
        actigraphy_df = pd.read_csv(f"{self.path}/data/filtered/{file}")
        # rename the axis to be timestamp, x, y, z
        actigraphy_df.rename(columns={'timestamp': 'timestamp', 'Accelerometer X': 'x', 'Accelerometer Y': 'y', 'Accelerometer Z': 'z'}, inplace=True)
        actigraphy_df = self.apply_filter_to_all_axis(actigraphy_df)
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
        # loggin the kurtosis of the x axis
        # print("Calculating the kurtosis of the x axis")
        # new_df['x_kurtosis'] = x_series.groupby(np.arange(len(x_series.index)) // 800).kurtosis()        
        # loggin the fft mean of the x axis
        print("Calculating the fft mean of the x axis")
        new_df['x_fft_mean'] = x_series.groupby(np.arange(len(x_series.index)) // 800).apply(self.get_freq_domain_features).apply(pd.Series).mean(axis=1)
        # loggin the fft std of the x axis
        print("Calculating the fft std of the x axis")
        new_df['x_fft_std'] = x_series.groupby(np.arange(len(x_series.index)) // 800).apply(self.get_freq_domain_features).apply(pd.Series).std(axis=1)
        # loggin the fft min of the x axis
        print("Calculating the fft min of the x axis")
        new_df['x_fft_min'] = x_series.groupby(np.arange(len(x_series.index)) // 800).apply(self.get_freq_domain_features).apply(pd.Series).min(axis=1)
        # loggin the fft max of the x axis
        print("Calculating the fft max of the x axis")
        new_df['x_fft_max'] = x_series.groupby(np.arange(len(x_series.index)) // 800).apply(self.get_freq_domain_features).apply(pd.Series).max(axis=1)
        # loggin the fft median of the x axis
        print("Calculating the fft median of the x axis")
        new_df['x_fft_median'] = x_series.groupby(np.arange(len(x_series.index)) // 800).apply(self.get_freq_domain_features).apply(pd.Series).median(axis=1)
        # loggin the fft skewness of the x axis
        print("Calculating the fft skewness of the x axis")
        new_df['x_fft_skew'] = x_series.groupby(np.arange(len(x_series.index)) // 800).apply(self.get_freq_domain_features).apply(pd.Series).skew(axis=1)
        # new_df['x_fft_kurtosis'] = x_series.groupby(np.arange(len(x_series.index)) // 800).apply(self.get_freq_domain_features).apply(pd.Series).kurtosis(axis=1)
        # generate features for y axis
        y_series = actigraphy_df['y']
        # loggin the mean of the y axis
        print("Calculating the mean of the y axis")
        new_df['y_mean'] = y_series.groupby(np.arange(len(y_series.index)) // 800).mean()
        # loggin the std of the y axis
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
        # loggin the kurtosis of the y axis
        # print("Calculating the kurtosis of the y axis")
        # new_df['y_kurtosis'] = y_series.groupby(np.arange(len(y_series.index)) // 800).kurtosis()
        # loggin the fft mean of the y axis
        print("Calculating the fft mean of the y axis")
        new_df['y_fft_mean'] = y_series.groupby(np.arange(len(y_series.index)) // 800).apply(self.get_freq_domain_features).apply(pd.Series).mean(axis=1)
        # loggin the fft std of the y axis
        print("Calculating the fft std of the y axis")
        new_df['y_fft_std'] = y_series.groupby(np.arange(len(y_series.index)) // 800).apply(self.get_freq_domain_features).apply(pd.Series).std(axis=1)
        # loggin the fft min of the y axis
        print("Calculating the fft min of the y axis")
        new_df['y_fft_min'] = y_series.groupby(np.arange(len(y_series.index)) // 800).apply(self.get_freq_domain_features).apply(pd.Series).min(axis=1)
        # loggin the fft max of the y axis
        print("Calculating the fft max of the y axis")
        new_df['y_fft_max'] = y_series.groupby(np.arange(len(y_series.index)) // 800).apply(self.get_freq_domain_features).apply(pd.Series).max(axis=1)
        # loggin the fft median of the y axis
        print("Calculating the fft median of the y axis")
        new_df['y_fft_median'] = y_series.groupby(np.arange(len(y_series.index)) // 800).apply(self.get_freq_domain_features).apply(pd.Series).median(axis=1)
        # loggin the fft skew of the y axis
        print("Calculating the fft skew of the y axis")
        new_df['y_fft_skew'] = y_series.groupby(np.arange(len(y_series.index)) // 800).apply(self.get_freq_domain_features).apply(pd.Series).skew(axis=1)
        # new_df['y_fft_kurtosis'] = y_series.groupby(np.arange(len(y_series.index)) // 800).apply(self.get_freq_domain_features).apply(pd.Series).kurtosis(axis=1)
        # generate features for z axis
        z_series = actigraphy_df['z']
        # loggin the mean of the z axis
        print("Calculating the mean of the z axis")
        new_df['z_mean'] = z_series.groupby(np.arange(len(z_series.index)) // 800).mean()
        # loggin the std of the z axis
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
        # # loggin the kurtosis of the z axis
        # print("Calculating the kurtosis of the z axis")
        # new_df['z_kurtosis'] = z_series.groupby(np.arange(len(z_series.index)) // 800).kurtosis()
        # loggin the fft mean of the z axis
        print("Calculating the fft mean of the z axis")
        new_df['z_fft_mean'] = z_series.groupby(np.arange(len(z_series.index)) // 800).apply(self.get_freq_domain_features).apply(pd.Series).mean(axis=1)
        # loggin the fft std of the z axis
        print("Calculating the fft std of the z axis")
        new_df['z_fft_std'] = z_series.groupby(np.arange(len(z_series.index)) // 800).apply(self.get_freq_domain_features).apply(pd.Series).std(axis=1)
        # loggin the fft min of the z axis
        print("Calculating the fft min of the z axis")
        new_df['z_fft_min'] = z_series.groupby(np.arange(len(z_series.index)) // 800).apply(self.get_freq_domain_features).apply(pd.Series).min(axis=1)
        # loggin the fft max of the z axis
        print("Calculating the fft max of the z axis")
        new_df['z_fft_max'] = z_series.groupby(np.arange(len(z_series.index)) // 800).apply(self.get_freq_domain_features).apply(pd.Series).max(axis=1)
        # loggin the fft median of the z axis
        print("Calculating the fft median of the z axis")
        new_df['z_fft_median'] = z_series.groupby(np.arange(len(z_series.index)) // 800).apply(self.get_freq_domain_features).apply(pd.Series).median(axis=1)
        # loggin the fft skew of the z axis
        print("Calculating the fft skew of the z axis")
        new_df['z_fft_skew'] = z_series.groupby(np.arange(len(z_series.index)) // 800).apply(self.get_freq_domain_features).apply(pd.Series).skew(axis=1)
        # new_df['z_fft_kurtosis'] = z_series.groupby(np.arange(len(z_series.index)) // 800).apply(self.get_freq_domain_features).apply(pd.Series).kurtosis(axis=1)
        #  save the new dataframe to csv
        if is_dominant_hand:
            new_df.to_csv(f'{self.path}/data/actigraphy_features/DS_{subject_id}_actigraphy_features_dominant.csv', index=False)
        else:
            new_df.to_csv(f'{self.path}/data/actigraphy_features/DS_{subject_id}_actigraphy_features_non_dominant.csv', index=False)
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

    def test_low_pass_filter(self, subject_id):
        '''
        This function takes in the subject_id, extract the first 10 minutes of data (80Hz) and plot the original and filtered data
        :params: subject_id: the id of the subject
        '''
        subject_file = self.get_file_name(subject_id=subject_id)[0]
        file_df = pd.read_csv(f"{self.path}/data/filtered/{subject_file}")
        # print out the columns
        # print(file_df.columns)
        # rename the axis to be timestamp, x, y, z
        file_df.rename(columns={'timestamp': 'timestamp', 'Accelerometer X': 'x', 'Accelerometer Y': 'y', 'Accelerometer Z': 'z'}, inplace=True)
        # get the first 30 seconds of data
        file_df = file_df.iloc[1000000:1000000 +30*80]
        # plot the original data for x axis in one subplot
        plt.subplot(2,1,1)
        plt.plot(file_df['x'])
        plt.title("Original data")
        # plot the filtered data for x axis in another subplot
        plt.subplot(2,1,2)
        plt.plot(self.low_pass_filter(file_df['x'], cutoff=4, fs=80, order=5))
        plt.title("Filtered data")
        plt.show()


accelSummary = ActigraphSummary(ROOT_DIR)
# accelSummary.get_file_name(10)
# accelSummary.print_head_file(10)
# accelSummary.test_low_pass_filter(10)
# for i in range(23, 33):
#     accelSummary.segment_and_add_features(i, is_dominant_hand=True)
#     accelSummary.segment_and_add_features(i, is_dominant_hand=False)

accelSummary.print_head_file(23)


# accelSummary.segment_and_add_features(10)
