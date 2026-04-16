#!/usr/bin/env python3

#### Calculates summary scores on PENN CNP measures

import os
import sys
import numpy as np
import pandas as pd  # Data management library
import shutil # for copying files
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# **Read in files**

# Read in PENN CNP
# Add path to where penn_cnp.csv is located
df_penn_tmp = pd.read_csv("penn_cnp.csv")
print(f"Read in {len(df_penn_tmp)} rows from penn_cnp.csv")

# Simplify session_num
df_penn_tmp['session_num'] = df_penn_tmp['session_num'].apply(lambda x: 'BAS' if 'BAS' in str(x) else x)

# Drop duplicate rows based on 'custom_ID'
df_penn = df_penn_tmp.drop_duplicates(subset=['custom_ID'])

# Print the count for each session in the penn data to verify filtering
print('Penn:\n', df_penn['session_num'].value_counts())

# Rename relevant columns
dict_penn = {'penncnp_0006': 'MPraxis_RT', # Motor Praxis
             'penncnp_0009': 'CPT_Num_TP', # Continuous Performance Test
             'penncnp_0010': 'CPT_Num_FP',
             'penncnp_0011': 'CPT_Num_TN',
             'penncnp_0012': 'CPT_Num_FN',
             'penncnp_0015': 'CPT_Let_TP',
             'penncnp_0016': 'CPT_Let_FP',
             'penncnp_0017': 'CPT_Let_TN',
             'penncnp_0018': 'CPT_Let_FN',
             'penncnp_0039': 'NB_0Corr', # N-Back
             'penncnp_0041': 'NB_0RT',
             'penncnp_0042': 'NB_1Corr',
             'penncnp_0044': 'NB_1RT',
             'penncnp_0045': 'NB_2Corr',
             'penncnp_0047': 'NB_2RT',
             'penncnp_0054': 'CET_Corr', # Conditional Exclusion Task
             'penncnp_0055': 'CET_RT',
             'penncnp_069': 'MEDF_Corr_happy', # Measured Emotion Differentiation Task
             'penncnp_0070': 'MEDF_Corr_sad',
             'penncnp_0071': 'MEDF_Corr_angry',
             'penncnp_0072': 'MEDF_Corr_fear',
             'penncnp_0073': 'MEDF_RT_happy',
             'penncnp_0074': 'MEDF_RT_sad',
             'penncnp_0075': 'MEDF_RT_angry',
             'penncnp_0076': 'MEDF_RT_fear',
             'penncnp_0100': 'CPW_TP', # Word Memory Test
             'penncnp_0101': 'CPW_TN',
             'penncnp_0102': 'CPW_FP',
             'penncnp_0103': 'CPW_FN',
             'penncnp_0108': 'CPW_Corr',
             'penncnp_0109': 'CPW_RT',
             # 'penncnp_0124': 'CTAP_MeanTaps_Dom', # Computerized Finger-Tapping Task - mostly missing
             'penncnp_0130': 'ER40_Corr', # Emotion Recognition Task
             'penncnp_0131': 'ER40_RT',
             'penncnp_0136': 'ER40_Corr_Anger',
             'penncnp_0137': 'ER40_Corr_Fear',
             'penncnp_0138': 'ER40_Corr_Happy',
             'penncnp_0139': 'ER40_Corr_NoEmot',
             'penncnp_0140': 'ER40_Corr_Sad',
             'penncnp_0156': 'ER40_RT_Anger',
             'penncnp_0157': 'ER40_RT_Fear',
             'penncnp_0158': 'ER40_RT_Happy',
             'penncnp_0159': 'ER40_RT_NoEmot',
             'penncnp_0160': 'ER40_RT_Sad',
             'penncnp_0196': 'VRT_PctCorr', # Verbal Reasoning Test
             'penncnp_0198': 'VRT_CorrRT',
             'penncnp_0202': 'FMT_TP', # (CPF) Face Memory Test
             'penncnp_0203': 'FMT_TN',
             'penncnp_0204': 'FMT_FP',
             'penncnp_0205': 'FMT_FN',
             'penncnp_0210': 'FMT_Corr',
             'penncnp_0211': 'FMT_RT',
             'penncnp_0226': 'VOLT_Corr', # Visual Object Learning Test
             'penncnp_0228': 'VOLT_RT', # for Corr responses
            }

# Rename columns according to dict_penn
df_penn_renamed = df_penn.rename(columns=dict_penn)

# Print list of renamed columns to verify
vars_list = list(dict_penn.values())
print("Renamed columns in PENN CNP data:")
print(vars_list)


def calc_sigproc_metrics(df, prefix):
    """
    Calculate signal detection theory metrics: d-prime, a-prime, and criterion-c
    
    Parameters:
    df: DataFrame containing the data
    prefix: String prefix for the columns (e.g., 'CPT_Num', 'CPT_Let', 'NB_0', etc.)
    
    Returns:
    Dictionary with calculated metrics for each subject
    """
    import numpy as np
    
    # Define column names based on prefix
    tp_col = f"{prefix}TP"  # True Positives (Hits)
    fp_col = f"{prefix}FP"  # False Positives (False Alarms)
    tn_col = f"{prefix}TN"  # True Negatives (Correct Rejections)
    fn_col = f"{prefix}FN"  # False Negatives (Misses)
    
    # Check if all required columns exist
    required_cols = [tp_col, fp_col, tn_col, fn_col]
    available_cols = [col for col in required_cols if col in df.columns]
    
    if len(available_cols) < 4:
        print(f"Warning: Not all required columns found for {prefix}")
        print(f"Available: {available_cols}")
        return None
    
    # Get the data
    tp = df[tp_col].values
    fp = df[fp_col].values
    tn = df[tn_col].values
    fn = df[fn_col].values
    
    # Calculate hit rate and false alarm rate
    hit_rate = tp / (tp + fn)  # P(Hit) = TP / (TP + FN)
    fa_rate = fp / (fp + tn)   # P(False Alarm) = FP / (FP + TN)
    
    # Apply corrections for extreme values (0 or 1) to avoid infinite z-scores
    # Using 1/(2N) and 1-1/(2N) correction (Macmillan & Creelman, 2005)
    n_signal = tp + fn      # Total signal trials
    n_noise = fp + tn       # Total noise trials
    
    # Correct hit rate
    hit_rate_corrected = np.where(hit_rate == 0, 1/(2*n_signal), hit_rate)
    hit_rate_corrected = np.where(hit_rate_corrected == 1, 1 - 1/(2*n_signal), hit_rate_corrected)
    
    # Correct false alarm rate
    fa_rate_corrected = np.where(fa_rate == 0, 1/(2*n_noise), fa_rate)
    fa_rate_corrected = np.where(fa_rate_corrected == 1, 1 - 1/(2*n_noise), fa_rate_corrected)
    
    # Calculate d-prime
    from scipy.stats import norm
    d_prime = norm.ppf(hit_rate_corrected) - norm.ppf(fa_rate_corrected)
    
    # Calculate a-prime (non-parametric sensitivity index)
    # A' = 0.5 + ((H - F)(1 + H - F)) / (4H(1 - F)) when H >= F
    # A' = 0.5 - ((F - H)(1 + F - H)) / (4F(1 - H)) when F > H
    a_prime = np.where(
        hit_rate >= fa_rate,
        0.5 + ((hit_rate - fa_rate) * (1 + hit_rate - fa_rate)) / (4 * hit_rate * (1 - fa_rate)),
        0.5 - ((fa_rate - hit_rate) * (1 + fa_rate - hit_rate)) / (4 * fa_rate * (1 - hit_rate))
    )
    
    # Calculate criterion c (response bias)
    # c = -0.5 * (Z(hit_rate) + Z(fa_rate))
    criterion_c = -0.5 * (norm.ppf(hit_rate_corrected) + norm.ppf(fa_rate_corrected))
    
    return {
        f'{prefix}d-prime': d_prime,
        f'{prefix}a-prime': a_prime,
        f'{prefix}criterion-c': criterion_c,
        f'{prefix}hit-rate': hit_rate,
        f'{prefix}fa-rate': fa_rate
    }


# Apply the function to calculate metrics for different tasks
print("Calculating signal detection metrics for PENN CNP tasks...")

# Find all unique prefixes that have TP, FP, TN, FN columns
all_cols = df_penn_renamed.columns
tp_cols = [col for col in all_cols if col.endswith('TP')]
prefixes = []

for tp_col in tp_cols:
    prefix = tp_col[:-2]  # Remove 'TP' suffix
    # Check if corresponding FP, TN, FN columns exist
    if all(f"{prefix}{suffix}" in all_cols for suffix in ['FP', 'TN', 'FN']):
        prefixes.append(prefix)

print(f"Found prefixes with complete TP/FP/TN/FN data: {prefixes}")

# Calculate metrics for each prefix
for prefix in prefixes:
    print(f"\nCalculating metrics for {prefix}...")

    metrics = calc_sigproc_metrics(df_penn_renamed, prefix)
    
    if metrics is not None:
        # Add the calculated metrics to the dataframe
        for metric_name, values in metrics.items():
            df_penn_renamed[metric_name] = values
        
        # Print summary statistics
        print(f"  d-prime: Mean = {np.nanmean(metrics[f'{prefix}d-prime']):.3f}, SD = {np.nanstd(metrics[f'{prefix}d-prime']):.3f}")
        print(f"  a-prime: Mean = {np.nanmean(metrics[f'{prefix}a-prime']):.3f}, SD = {np.nanstd(metrics[f'{prefix}a-prime']):.3f}")
        print(f"  criterion-c: Mean = {np.nanmean(metrics[f'{prefix}criterion-c']):.3f}, SD = {np.nanstd(metrics[f'{prefix}criterion-c']):.3f}")

# Show the new columns created
new_metric_cols = [
    col for col in df_penn_renamed.columns
    if any(suffix in col for suffix in ['d-prime', 'a-prime', 'criterion-c'])
]
print(f"\nNew metric columns created: {new_metric_cols}")


# Calculate combined CPT metrics by pooling Num and Let conditions
print("\nCalculating combined CPT metrics (Num + Let conditions)...")

# Check if both CPT_Num and CPT_Let data are available
cpt_num_cols = ['CPT_Num_TP', 'CPT_Num_FP', 'CPT_Num_TN', 'CPT_Num_FN']
cpt_let_cols = ['CPT_Let_TP', 'CPT_Let_FP', 'CPT_Let_TN', 'CPT_Let_FN']

if all(col in df_penn_renamed.columns for col in cpt_num_cols + cpt_let_cols):
    # Combine TP, FP, TN, FN across Num and Let conditions
    df_penn_renamed['CPT_all_TP'] = df_penn_renamed['CPT_Num_TP'] + df_penn_renamed['CPT_Let_TP']
    df_penn_renamed['CPT_all_FP'] = df_penn_renamed['CPT_Num_FP'] + df_penn_renamed['CPT_Let_FP']
    df_penn_renamed['CPT_all_TN'] = df_penn_renamed['CPT_Num_TN'] + df_penn_renamed['CPT_Let_TN']
    df_penn_renamed['CPT_all_FN'] = df_penn_renamed['CPT_Num_FN'] + df_penn_renamed['CPT_Let_FN']
    
    # Calculate combined metrics using the existing function
    cpt_all_metrics = calc_sigproc_metrics(df_penn_renamed, 'CPT_all_')
    
    if cpt_all_metrics is not None:
        # Add the calculated metrics to the dataframe
        for metric_name, values in cpt_all_metrics.items():
            df_penn_renamed[metric_name] = values
        
        # Print summary statistics for combined CPT
        print(f"  Combined CPT d-prime: Mean = {np.nanmean(cpt_all_metrics['CPT_all_d-prime']):.3f}, SD = {np.nanstd(cpt_all_metrics['CPT_all_d-prime']):.3f}")
        print(f"  Combined CPT a-prime: Mean = {np.nanmean(cpt_all_metrics['CPT_all_a-prime']):.3f}, SD = {np.nanstd(cpt_all_metrics['CPT_all_a-prime']):.3f}")
        print(f"  Combined CPT criterion-c: Mean = {np.nanmean(cpt_all_metrics['CPT_all_criterion-c']):.3f}, SD = {np.nanstd(cpt_all_metrics['CPT_all_criterion-c']):.3f}")
        
        # Update the new_metric_cols list to include the combined metrics
        cpt_combined_cols = ['CPT_all_d-prime', 'CPT_all_a-prime', 'CPT_all_criterion-c', 'CPT_all_hit-rate', 'CPT_all_fa-rate']
        new_metric_cols.extend(cpt_combined_cols)
        
        print(f"\nCombined CPT metrics created: {cpt_combined_cols}")
        
        # Show sample sizes for comparison
        n_valid_num = df_penn_renamed['CPT_Num_TP'].notna().sum()
        n_valid_let = df_penn_renamed['CPT_Let_TP'].notna().sum()
        n_valid_combined = df_penn_renamed['CPT_all_TP'].notna().sum()
        
        print(f"\nSample sizes:")
        print(f"  CPT Num: {n_valid_num}")
        print(f"  CPT Let: {n_valid_let}")
        print(f"  CPT Combined: {n_valid_combined}")
        
    else:
        print("Error calculating combined CPT metrics")
        
else:
    missing_cols = [col for col in cpt_num_cols + cpt_let_cols if col not in df_penn_renamed.columns]
    print(f"Cannot calculate combined CPT metrics. Missing columns: {missing_cols}")


# Get the list of renamed variables for analysis
penn_vars = vars_list + new_metric_cols + ['CPT_all_d-prime', 'CPT_all_a-prime', 'CPT_all_criterion-c']

# Check which variables are actually present in the dataframe
available_vars = [var for var in penn_vars if var in df_penn_renamed.columns]
missing_vars = [var for var in penn_vars if var not in df_penn_renamed.columns]

print(f"\nAvailable variables ({len(penn_vars)}): {penn_vars}")
if missing_vars:
    print(f"Missing variables ({len(missing_vars)}): {missing_vars}")

# Save the processed PENN CNP data to a CSV file
# Edit path to where file should be saved
output_file = "PENN_CNP_processed.csv"
df_penn_renamed.to_csv(output_file, index=False)