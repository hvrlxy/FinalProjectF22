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
        self.df = self.apply_filter_to_all_axis()
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
        # gểnate features for x axis
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
        y_series = self.df['y']
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
        z_series = self.df['z']
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
        return new_df

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