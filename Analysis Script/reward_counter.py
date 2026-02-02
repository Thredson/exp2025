# -*- coding: utf-8 -*-
import json
import pandas as pd
import glob
import os
from scipy import stats

# Get all JSON files in the folder selected
target_folder = 'feature_data'
json_files = glob.glob(f'{target_folder}/*.json')

reward_strings = []

reward_list = []

for file_path in json_files:
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            
        df = pd.DataFrame(data)
        
        df_trials = df.dropna(subset=['trial_number'])
        
        pro_ID = list(df.loc[3]['response'].values())[0]
        
        true_count = df_trials['reward'].sum()

        reward_list.append(true_count)

        reward_string = pro_ID + ',' + str(true_count * 0.01)
        reward_strings.append(reward_string)
        
        print(f"Processed {os.path.basename(file_path)}: {reward_string}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        continue

mu_target_r = 64.035
mu_target_q = 63.61
mu_target_b = 68.81
t_stat_r, p_val_r = stats.ttest_1samp(reward_list, popmean=mu_target_r)
t_stat_q, p_val_q = stats.ttest_1samp(reward_list, popmean=mu_target_q)
t_stat_b, p_val_b = stats.ttest_1samp(reward_list, popmean=mu_target_b)

print(f"t-test between random and target")
print(f"t-statistic: {t_stat_r}")
print(f"p-value: {p_val_r}")

print(f"t-test between q-learning and target")
print(f"t-statistic: {t_stat_q}")
print(f"p-value: {p_val_q}")

print(f"t-test between benchmark and target")
print(f"t-statistic: {t_stat_b}")
print(f"p-value: {p_val_b}")

# # Write all reward strings to output file
# output_file = f'{target_folder}/reward_results.txt'
# with open(output_file, 'w') as f:
#     for reward_string in reward_strings:
#         f.write(reward_string + '\n')

# print(f"\nAll results saved to {output_file}")
# print(f"Total files processed: {len(reward_strings)}")