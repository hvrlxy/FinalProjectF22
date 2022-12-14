import os, sys
import scipy
from scipy.signal import find_peaks
from scipy import stats
import pandas as pd
import numpy as np
import datetime
import glob
import warnings

class ActigraphSummary:
    def __init__(self, raw_df):
        self.df = raw_df 

    def apply_filter_to_all_axis(self):
        '''
        This function takes in the subject's id and apply the low pass filter to all the axis. This function return the filtered data.
        :params: subject_id: id of the subject
        :params: self.df: the actigraphy dataframe
        :return: filtered data
        '''
        # rename the axis to be timestamp, x, y, z
        self.df.rename(columns={'timestamp': 'timestamp', 'Accelerometer X': 'x', 'Accelerometer Y': 'y', 'Accelerometer Z': 'z'}, inplace=True)
        # apply low pass filter to all the axis
        self.df['x'] = self.moving_average_filtered(self.df['x'])
        self.df['y'] = self.moving_average_filtered(self.df['y'])
        self.df['z'] = self.moving_average_filtered(self.df['z'])

    def segment_and_add_features(self):
        '''
        This function takes in the subject's id, segment the data into 10s window and add features to the data.
        Return a new dataframe with the features added and the timestamp of the first sample in the segment
        :params: subject_id: id of the subject
        :params: is_dominant_hand: whether the data we are looking at is dominant/right hand or not
        :return: a new dataframe with the features added and the timestamp of the first sample in the segment
        '''
        # rename the axis to be timestamp, x, y, z
        self.df.rename(columns={'timestamp': 'timestamp', 'Accelerometer X': 'x', 'Accelerometer Y': 'y', 'Accelerometer Z': 'z'}, inplace=True)
        self.apply_filter_to_all_axis()
        # loop through the data in 10s window (80Hz * 10s = 800 samples)
        window_size = 800
        length_data = len(self.df)

        # create a new dataframe to store the features
        new_df = pd.DataFrame()
        timestamp = []
        for i in range(0, length_data, window_size):
            timestamp.append(self.df['timestamp'][i])
        # print the first and last timestamp of the actigraphy data
        print(f"First timestamp: {self.df['timestamp'][0]}")
        print(f"Last timestamp: {self.df['timestamp'][length_data-1]}")
        print("Length of the actigraphy data: ", length_data)
        new_df['timestamp'] = timestamp
        # g???nate features for x axis
        x_series = self.df['x']
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
        y_series = self.df['y']
        print("Calculating the mean of the y axis")
        new_df['y_mean'] = self.df['y'].groupby(np.arange(len(self.df.index)) // 800).mean()
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
        z_series = self.df['z']
        print("Calculating the mean of the z axis")
        new_df['z_mean'] = z_series.groupby(np.arange(len(self.df.index)) // 800).mean()
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
        
        return new_df
    
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