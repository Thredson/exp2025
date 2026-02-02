# -*- coding: utf-8 -*-
import json
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import chi2_contingency

# Get all JSON files in the folder selected

def extract_folder_data(target_folder):
    json_files = glob.glob(f'{target_folder}/*.json')
    
    choice_count = {}
    
    note_count = {}
    
    reward_count = []
    
    reward_count_total = []
    
    # demographics
    
    ages = []
    genders = []
    education_levels = []
    
    
    for file_path in json_files:
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                
            df = pd.DataFrame(data)
            
            
            # analyzing multiple choice question distribution
            
            df_choice = df[df['trial_type'] == 'survey-multi-choice']
            
            choice = list(df_choice["response"].iloc[0].values())[0]
            
            note = list(df_choice["response"].iloc[0].values())[1]
            
            choice_count[choice] = choice_count.get(choice, 0) + 1
            
            note_count[note] = note_count.get(note, 0) + 1
            
            df_demographics = df[df['trial_type'] == 'survey-text'].iloc[-1:]
            if not df_demographics.empty:
                demo_response = df_demographics['response'].iloc[0]
                
                # Extract age, gender, and education
                if 'age' in demo_response:
                    ages.append(int(demo_response['age']))
                if 'gender' in demo_response:
                    genders.append(demo_response['gender'])
                if 'education' in demo_response:
                    education_levels.append(demo_response['education'])    
            
            df_trials = df.dropna(subset=['trial_number'])
            
            df_trials_test = df_trials[df_trials["phase"] == "testing"]
            
            true_count = df_trials_test['reward'].sum() 
            
            true_count_total = df_trials['reward'].sum() 
            
            
            if note != 'no':
                pro_ID = list(df.loc[3]['response'].values())[0]
                print(f'{pro_ID} took notes and got {true_count} correct')
                print(f'he though the pattern is {choice}')
                print(f'{file_path}')
            
                    
            reward_count_total.append(true_count_total)
            
            reward_count.append(true_count)
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue
        
    choice_count['feature dependent'] = choice_count.pop('Pictures with some specific features should be preferred',0)
    choice_count['PD rule'] = choice_count.pop('The picture you picked previously should not be picked on the next occasion, which means the picture not picked previously should now be picked')
    choice_count['fixed ranks'] = choice_count.pop('There is a ranking between six pictures. For example, Picture 1 > Picture 2 > Picture 3... A picture with a relative higher rank in the present pair should be preferred.')
    choice_count['random'] = choice_count.pop('Completely random')

    # Create demographics dictionary
    demographics = {
        'ages': ages,
        'genders': genders,
        'education': education_levels
    }
    
    return choice_count, note_count, reward_count, reward_count_total, demographics
    
print("Extracting feature_data...")
feature_choice, feature_notes, feature_rewards, feature_rewards_total, feature_demographics = extract_folder_data('feature_data')

print("\nExtracting non_feature_data...")
non_feature_choice, non_feature_notes, non_feature_rewards, non_feature_rewards_total, non_feature_demographics = extract_folder_data('non_feature_data')

print("\n" + "="*60)
print("DEMOGRAPHICS ANALYSIS")
print("="*60)

# Feature group demographics
print("\nFeature Group Demographics:")
print("-" * 30)
print(f"Sample size: {len(feature_demographics['ages'])}")
print(f"Age: Mean = {np.mean(feature_demographics['ages']):.1f}, SD = {np.std(feature_demographics['ages']):.1f}")
print(f"     Range = {min(feature_demographics['ages'])} - {max(feature_demographics['ages'])}")

# Gender distribution
gender_counts_feature = pd.Series(feature_demographics['genders']).value_counts()
print("\nGender distribution:")
for gender, count in gender_counts_feature.items():
    print(f"  {gender}: {count} ({count/len(feature_demographics['genders'])*100:.1f}%)")

# Education distribution
edu_counts_feature = pd.Series(feature_demographics['education']).value_counts()
print("\nEducation level:")
for edu, count in edu_counts_feature.items():
    print(f"  {edu}: {count} ({count/len(feature_demographics['education'])*100:.1f}%)")

# Non-feature group demographics
print("\nNon-Feature Group Demographics:")
print("-" * 30)
print(f"Sample size: {len(non_feature_demographics['ages'])}")
print(f"Age: Mean = {np.mean(non_feature_demographics['ages']):.1f}, SD = {np.std(non_feature_demographics['ages']):.1f}")
print(f"     Range = {min(non_feature_demographics['ages'])} - {max(non_feature_demographics['ages'])}")

# Gender distribution
gender_counts_non_feature = pd.Series(non_feature_demographics['genders']).value_counts()
print("\nGender distribution:")
for gender, count in gender_counts_non_feature.items():
    print(f"  {gender}: {count} ({count/len(non_feature_demographics['genders'])*100:.1f}%)")

# Education distribution
edu_counts_non_feature = pd.Series(non_feature_demographics['education']).value_counts()
print("\nEducation level:")
for edu, count in edu_counts_non_feature.items():
    print(f"  {edu}: {count} ({count/len(non_feature_demographics['education'])*100:.1f}%)")


'''    
plt.figure(figsize=(8, 6))
sns.kdeplot(reward_count, fill=True)
plt.hist(reward_count, bins=20, density=True, alpha=0.7)
plt.show()
'''

target_folder = 'non_feature_data'      
reward_count = non_feature_rewards    
reward_count_total = non_feature_rewards_total
choice_count = non_feature_choice

df_plot = pd.DataFrame({'values': reward_count})
df_plot_total = pd.DataFrame({'values': reward_count_total})

mean_val = np.mean(reward_count)
median_val = np.median(reward_count)
std_val = np.std(reward_count)

plt.figure(figsize=(10, 6))
sns.histplot(df_plot['values'], kde=True, stat='density')
plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
plt.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
plt.title(f'Distribution of {target_folder}(μ={mean_val:.2f}, σ={std_val:.2f})')
plt.xlabel('Values')
plt.ylabel('Density')
plt.legend()
plt.savefig(f'{target_folder}/plot_test.png', 
            dpi=300,           # High resolution
            bbox_inches='tight', # Remove extra whitespace
            facecolor='white',   # Background color
            edgecolor='none')
plt.show()
plt.close()

mean_val_total = np.mean(reward_count_total)
median_val_total = np.median(reward_count_total)
std_val_total = np.std(reward_count_total)

plt.figure(figsize=(10, 6))
sns.histplot(df_plot_total['values'], kde=True, stat='density')
plt.axvline(mean_val_total, color='red', linestyle='--', label=f'Mean: {mean_val_total:.2f}')
plt.axvline(median_val_total, color='green', linestyle='--', label=f'Median: {median_val_total:.2f}')
plt.title(f'Distribution of {target_folder}(μ={mean_val_total:.2f}, σ={std_val_total:.2f})')
plt.xlabel('Values')
plt.ylabel('Density')
plt.legend()
plt.savefig(f'{target_folder}/plot_total.png', 
            dpi=300,           # High resolution
            bbox_inches='tight', # Remove extra whitespace
            facecolor='white',   # Background color
            edgecolor='none')
plt.show()
plt.close()

print("\n" + "="*60)
print("CHI-SQUARE TEST COMPARING FEATURE vs NON-FEATURE GROUPS")
print("="*60)

# Make sure both have all categories
all_choices = ['feature dependent', 'PD rule', 'fixed ranks', 'random']

# Create the contingency table from your choice_count values
observed_values = []
for choice in all_choices:
    observed_values.append([
        feature_choice.get(choice, 0),
        non_feature_choice.get(choice, 0)
    ])

# Convert to numpy array for chi-square test
contingency_table = np.array(observed_values)

# Display the contingency table
print("\nContingency Table:")
print(pd.DataFrame(contingency_table, 
                  index=all_choices,
                  columns=['Feature Group', 'Non-Feature Group']))

# Perform chi-square test
chi2, p_value, dof, expected_freq = chi2_contingency(contingency_table)

print("\nResults:")
print(f"Chi-square statistic: {chi2:.4f}")
print(f"p-value: {p_value:.6f}")
print(f"Degrees of freedom: {dof}")

if p_value < 0.05:
    print("\n✓ SIGNIFICANT RESULT (p < 0.05)")
    print("This means the two groups have significantly different response patterns.")
    print("The feature manipulation IS affecting how participants perceive patterns.")
else:
    print("\n✗ NOT SIGNIFICANT (p ≥ 0.05)")
    print("The two groups show similar distribution of perceived patterns.")

# Specific focus on "PD rule"
print("\n" + "="*60)
print("PD rule COMPARISON")
print("="*60)
feature_total = sum(feature_choice.values())
non_feature_total = sum(non_feature_choice.values())

feature_reverse_pct = (feature_choice.get('PD rule', 0) / feature_total * 100) if feature_total > 0 else 0
non_feature_reverse_pct = (non_feature_choice.get('PD rule', 0) / non_feature_total * 100) if non_feature_total > 0 else 0

print(f"Feature Group: {feature_choice.get('PD rule', 0)}/{feature_total} = {feature_reverse_pct:.1f}%")
print(f"Non-Feature Group: {non_feature_choice.get('PD rule', 0)}/{non_feature_total} = {non_feature_reverse_pct:.1f}%")

if abs(feature_reverse_pct - non_feature_reverse_pct) > 10:
    print(f"Notable difference of {abs(feature_reverse_pct - non_feature_reverse_pct):.1f} percentage points")



categories = list(choice_count.keys())
counts = list(choice_count.values())

df_choice_vis = pd.DataFrame(list(choice_count.items()), columns=['Answers', 'Count'])

plt.figure(figsize=(10, 6))

ax = sns.barplot(data=df_choice_vis, x='Answers', y='Count', palette='viridis')

for i, v in enumerate(df_choice_vis['Count']):
    ax.text(i, v + 0.2, str(v), ha='center', va='bottom', fontweight='bold', fontsize=12)
 
ax.set_yticks(range(0, max(df_choice_vis['Count']) + 3, 2))

plt.title('', fontsize=16)
plt.xlabel('Answers', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.savefig('bar_chart.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()  

plt.figure(figsize=(8, 8))
plt.pie(df_choice_vis['Count'], labels=df_choice_vis['Answers'], autopct='%1.1f%%', startangle=90)
plt.title('Count of Each Answer - Pie Chart')
plt.savefig('pie_chart.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()


















