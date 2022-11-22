import os, sys
import pandas as pd
from evaluation import *

ROOT_DIR = os.path.dirname(os.path.abspath(__file__), '../../..')

class GradientAscent:
    '''
    This class is used to run gradient ascent algrithm to find the optimal parameters for the algorithm.
    The paramter we will try to optimise is :
    - soft_threshold
    - hard_threshold
    - wake_score_threshold
    and the loss function will be prediction + correct_prompt_rate
    '''
    def __init__(self, soft_threshold, hard_threshold, wake_score_threshold, activation_function, wake_window = 10, motion_window = 3, motion_score_threshold = 0.5):
        self.soft_threshold = soft_threshold
        self.hard_threshold = hard_threshold
        self.wake_window = wake_window
        self.motion_window = motion_window
        self.wake_score_threshold = wake_score_threshold
        self.motion_score_threshold = motion_score_threshold
        self.activation_function = activation_function

        

    def loss_function(self):
        os.system(f"python3 {ROOT_DIR}/code/HINF/algorithm/test.py {self.soft_threshold} {self.hard_threshold} {self.wake_window} {self.motion_window} {self.wake_score_threshold} {self.motion_score_threshold} {self.activation_function}")
        accuracy, precision, recall, f1, fpr = compute_vanilla_metrics_for_subjects(ids=[i for i in range(10, 20)], is_dominant_hand=True)
        correct_prompt_rate = compute_correct_prompting_rate(ids=[i for i in range(10, 20)], is_dominant_hand=True, window_size=10)
        return (precision + correct_prompt_rate) / 2

    def gradient_ascent(self, learning_rate = 0.1, iterations = 1000, stop_threshold = 0.0001):
        for i in range(iterations):
            loss = self.loss_function()
            print(f"loss: {loss}")
            self.soft_threshold += learning_rate
            self.hard_threshold += learning_rate
            self.wake_score_threshold += learning_rate