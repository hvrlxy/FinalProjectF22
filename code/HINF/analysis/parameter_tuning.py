import os, sys
import pandas as pd
from evaluation import *

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/../../..'
print(ROOT_DIR)

soft_threshold_lst = [i * 100 for i in range(1, 10)]
hard_threshold_lst = [i * 100 for i in range(1, 10)]

wake_score_threshold_lst = [i * 0.1 for i in range(5, 10)]
activation_function_lst = ['sigmoid', 'linear']
            
def calculate_objective_function(soft_threshold, hard_threshold, wake_score_threshold, activation_function, wake_window=10, motion_window = 3, motion_score_threshold = 0.5 ):
    if soft_threshold > hard_threshold - 200:
        return -1
    os.system(f"python3 {ROOT_DIR}/code/HINF/algorithm/test.py {soft_threshold} {hard_threshold} {wake_window} {motion_window} {wake_score_threshold} {motion_score_threshold} {activation_function}")
    accuracy, precision, recall, f1, fpr = compute_vanilla_metrics_for_subjects(ids=[i for i in range(10, 20)], is_dominant_hand=True)
    correct_prompt_rate = compute_correct_prompting_rate(ids=[i for i in range(10, 20)], is_dominant_hand=True, window_size=10)
    return (precision + correct_prompt_rate) / 2

# create a dataframe that stores each of the soft_threshold, hard_threshold, wake_score_threshold pair
pd = pd.DataFrame(columns=['activation_function', 'wake_score_threshold', 'soft_threshold', 'hard_threshold', 'objective_values'])

for fn in activation_function_lst:
    for wst in wake_score_threshold_lst:
        for st in soft_threshold_lst:
            for ht in hard_threshold_lst:
                objective_value = calculate_objective_function(st, ht, wst, fn)
                pd = pd.append({'activation_function': fn, 'wake_score_threshold': wst, 'soft_threshold': st, 'hard_threshold': ht, 'objective_values': objective_value}, ignore_index=True)
                
pd.to_csv('parameter_tuning.csv')