# -*- coding: utf-8 -*-
import json

import pandas as pd

average = 0

for i in range(1, 1001):
    with open(f'agent_perfect/perfect_agent_{i:03d}.json', 'r') as file:
        data = json.load(file)
        
    df = pd.DataFrame(data)
    # print(df)

    df_trials = df.dropna(subset=['trial_number'])
    true_count = df_trials['reward'].sum()
    # print(f"Total rewards: {true_count}")
    average += true_count

average = average / 1000
average_perfect = average

average = 0

for i in range(1, 1001):
    with open(f'agent_random/random_agent_{i:03d}.json', 'r') as file:
        data = json.load(file)
        
    df = pd.DataFrame(data)
    # print(df)

    df_trials = df.dropna(subset=['trial_number'])
    true_count = df_trials['reward'].sum()
    # print(f"Total rewards: {true_count}")
    average += true_count

average = average / 1000
average_random = average 

average = 0

for i in range(1, 1001):
    with open(f'agent_Q/q_agent_{i:03d}.json', 'r') as file:
        data = json.load(file)
        
    df = pd.DataFrame(data)
    # print(df)

    df_trials = df.dropna(subset=['trial_number'])
    true_count = df_trials['reward'].sum()
    # print(f"Total rewards: {true_count}")
    average += true_count

average = average / 1000
average_q = average 

print(f"Average rewards for Benchmark agent: {average_perfect}")
print(f"Average rewards for Random agent: {average_random}")
print(f"Average rewards for Q-Learning agent: {average_q}")
# # Read from file
# with open('agent_perfect/perfect_agent_001.json', 'r') as file:
#     data = json.load(file)
    
# df = pd.DataFrame(data)
# # print(df)

# df_trials = df.dropna(subset=['trial_number'])

# true_count = df_trials['reward'].sum()

# print(f"Total rewards: {true_count}")

