import pandas as pd
import numpy as np
import datetime as dt
import os, sys
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as subplots

class DoubleThreshold:
    def __init__(self, soft_threshold, hard_threshold, wake_window, motion_window, 
                wake_score_threshold, motion_score_threshold, 
                activation_function = "linear"):
        self.soft_threshold = soft_threshold
        self.hard_threshold = hard_threshold
        self.wake_window = wake_window
        self.motion_window = motion_window
        self.wake_score_threshold = wake_score_threshold
        self.motion_score_threshold = motion_score_threshold
        self.activation_function = activation_function

    def calculate_score(self, auc_df):
        """
        if the AUC_sum is smaller than soft_threshold, then the score is 0
        if the AUC_sum is larger than hard_threshold, then the score is 1
        if the AUC_sum is between soft_threshold and hard_threshold, then the score is linearly interpolated
        or if activation_function is "sigmoid", then the score is sigmoid interpolated
        """
        # calculate the score
        if self.activation_function == "linear":
            auc_df["score"] = auc_df["AUC_sum"].apply(lambda x: self.linear_activation(x))
        elif self.activation_function == "sigmoid":
            auc_df["score"] = auc_df["AUC_sum"].apply(lambda x: self.sigmoid_activation(x))
        else:
            raise ValueError("activation_function must be either 'linear' or 'sigmoid'")
        return auc_df

    def linear_activation(self, x):
        if x <= self.soft_threshold:
            return 0
        elif x >= self.hard_threshold:
            return 1
        else:
            return (x - self.soft_threshold) / (self.hard_threshold - self.soft_threshold)
    
    def sigmoid_activation(self, x):
        if x <= self.soft_threshold:
            return 0
        elif x >= self.hard_threshold:
            return 1
        else:
            # map x to [-1, 1]
            x = 2 * (x - self.soft_threshold) / (self.hard_threshold - self.soft_threshold) - 1
            return 1 / (1 + np.exp(-x))

    def moving_average(self, x, w):
        seris = np.convolve(x, np.ones(w), 'valid') / w
        # add w-1 NaN to the front
        return np.concatenate((np.full(w-1, np.nan), seris))

    def detect(self, auc_df, plot = False, plot_path = None):
        """
        detect the wake and motion events
        """
        # calculate the score
        auc_df = self.calculate_score(auc_df)
        # calculate the moving average with the wake_window
        auc_df["moving_average"] = self.moving_average(auc_df["score"], self.wake_window)
        # calculate the moving average with the motion_window
        auc_df["moving_average_motion"] = self.moving_average(auc_df["score"], self.motion_window)
        # detect the wake events
        wake_events, wake_results_df = self.detect_wake(auc_df)
        # merge wake events that are within 5 minutes
        wake_events = self.merge_motion_events(wake_events)
        # detect the motion events
        motion_events, motion_results_df = self.detect_motion(auc_df)
        # merge the motion events
        motion_events = self.merge_motion_events(motion_events)

        if plot:
            self.generate_plot(auc_df, wake_events, motion_events, plot_path)
        return wake_events, motion_events, wake_results_df, motion_results_df

    def detect_wake(self, auc_df):
        '''
        detect the wake events by comparing the moving average with the wake_score_threshold.
        Once a wake event is detected, every timestamps beyond that point will be within the wake event
        '''
        wake_events = []
        wake_start = None
        for i in range(len(auc_df)):
            if auc_df["moving_average"][i] >= self.wake_score_threshold:
                if wake_start is None:
                    wake_start = i
            else:
                if wake_start is not None:
                    wake_end = i
                    wake_events.append([auc_df["timestamp"][wake_start], auc_df["timestamp"][wake_end]])
                    wake_start = None

        wake_results_df = pd.DataFrame()
        #append the timestamp in the auc_df to the wake_results_df
        wake_results_df["timestamp"] = auc_df["timestamp"]
        # convert the timestamp to datetime from epoch
        wake_results_df["datetime"] = wake_results_df["timestamp"].apply(lambda x: dt.datetime.fromtimestamp(x))   
        # only get the date
        wake_results_df["date"] = wake_results_df["datetime"].apply(lambda x: x.date())
        # remove the datetime
        wake_results_df = wake_results_df.drop(columns = ["datetime"])
        # create a column is_wake to indicate whether the timestamp is within a wake event
        wake_results_df["is_wake"] = auc_df["moving_average"].apply(lambda x: 1 if x >= self.wake_score_threshold else 0)
        return wake_events, wake_results_df
                
        

    def detect_motion(self, auc_df):
        '''
        detect the motion events by comparing the moving average with the motion_score_threshold
        '''
        motion_events = []
        motion_start = None
        for i in range(len(auc_df)):
            if auc_df["moving_average_motion"][i] >= self.motion_score_threshold:
                if motion_start is None:
                    motion_start = i
            else:
                if motion_start is not None:
                    motion_end = i
                    motion_events.append([auc_df["timestamp"][motion_start], auc_df["timestamp"][motion_end]])
                    motion_start = None

        motion_results_df = pd.DataFrame()
        #append the timestamp in the auc_df to the motion_results_df
        motion_results_df["timestamp"] = auc_df["timestamp"]
        # create a column is_motion to indicate whether the timestamp is within a motion event
        motion_results_df["is_motion"] = auc_df["moving_average"].apply(lambda x: 1 if x >= self.motion_score_threshold else 0)
        return motion_events, motion_results_df

    def merge_motion_events(self, motion_events):
        '''
        merge the motion events that are within 5 minutes
        '''
        merged_motion_events = []
        for i in range(len(motion_events)):
            if i == 0:
                merged_motion_events.append(motion_events[i])
            else:
                if motion_events[i][0] - merged_motion_events[-1][1] <= 5 * 60:
                    merged_motion_events[-1][1] = motion_events[i][1]
                else:
                    merged_motion_events.append(motion_events[i])
        return merged_motion_events

    def generate_plot(self, auc_df, wake_events, motion_events, plot_path):
        # create 4 subplots
        fig = subplots.make_subplots(rows = 4, cols = 1, shared_xaxes = True, vertical_spacing = 0.05)
        # resize the plot to 1200 x 2000
        fig.update_layout(height=1200, width=2000)
        # convert timestamp to datetime
        auc_df["timestamp"] = auc_df["timestamp"].apply(lambda x: dt.datetime.fromtimestamp(x))
        # add time of the day to the dataframe
        auc_df["time"] = auc_df["timestamp"].apply(lambda x: x.time())
        # sort by datetime
        auc_df = auc_df.sort_values(by = "timestamp")

        # plot the AUC_sum tp the first subplot
        fig.add_trace(go.Scatter(x=auc_df['time'], y=auc_df['AUC_sum'], name='AUCsum'), row=1, col=1)
        # plot the score to the second subplot
        fig.add_trace(go.Scatter(x=auc_df['time'], y=auc_df['score'], name='score'), row=2, col=1)
        # plot the moving average of the wake window to the third subplot
        fig.add_trace(go.Scatter(x=auc_df['time'], y=auc_df['moving_average'], name='moving_average'), row=3, col=1)
        # plot the moving average of the motion window to the fourth subplot
        fig.add_trace(go.Scatter(x=auc_df['time'], y=auc_df['moving_average_motion'], name='moving_average_motion'), row=4, col=1)

        # get the last timestamp of the dataframe
        # add the motion events to the fourth subplot with a vrect
        for wake_event in wake_events:
            fig.add_vrect(x0=dt.datetime.fromtimestamp(wake_event[0]).time(), x1=dt.datetime.fromtimestamp(wake_event[1]).time(), row=1, col=1, fillcolor="red", opacity=0.25, line_width=0)

        # add the motion events to the fourth subplot with a vrect
        for motion_event in motion_events:
            fig.add_vrect(x0=dt.datetime.fromtimestamp(motion_event[0]).time(), x1=dt.datetime.fromtimestamp(motion_event[1]).time(), row=1, col=1, fillcolor="green", opacity=0.25, line_width=0)

        # add a line for the wake score threshold to the third subplot
        fig.add_hline(y=self.wake_score_threshold, row=3, col=1, line_dash="dash", line_color="red")
        # add a line for the motion score threshold to the fourth subplot
        fig.add_hline(y=self.motion_score_threshold, row=4, col=1, line_dash="dash", line_color="green")
        # save the plot
        fig.write_image(plot_path)