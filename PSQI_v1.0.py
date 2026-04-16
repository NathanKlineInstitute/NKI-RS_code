#!/usr/bin/env python3

# ### Recalculates sleep score from PSQI

import os
import sys
import numpy as np
import pandas as pd  # Data management library
import matplotlib.pyplot as plt  # graphics
import statsmodels.api as sm
import seaborn as sns  # graphics ++
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import scipy.stats as stats


# Read in PSQI
# Edit file path as needed
df_PSQI = pd.read_csv("psqi.csv")
print(f'total number of rows in df_PSQI: {len(df_PSQI)}.')


# Get rid of '-1' from column names
df_PSQI.columns = [col[:-2] if col.startswith('psqi_') and col.endswith('-1') else col for col in df_PSQI.columns]

# Select columns of PSQI
psqi_columns = [col for col in df_PSQI.columns if col.startswith('psqi')]

for col in psqi_columns:
    if not col in ['psqi_01', 'psqi_03']:  # 1 and 3 are in 24-hr time format
        # Replace MD and DK with NaN
        df_PSQI[col] = df_PSQI[col].replace({'MD': np.nan, 'DK': np.nan})


# ## Calculates difference in time spent in bed based on reported time to bed and time waking up
# Difference in time was calculated manually instead of using datetime tools

# Extract hours and minutes
def extract_time_components(time_str):
    if isinstance(time_str, str) and time_str.strip():
        try:
            parts = time_str.split(':')
            if len(parts) >= 2:
                hours = int(parts[0])
                minutes = int(parts[1])
                return hours, minutes
        except (ValueError, IndexError):
            pass
    return np.nan, np.nan

# Calculate time in bed without filtering
def hrs_in_bed_time(bed_hour, bed_minute, wake_hour, wake_minute):
    if (bed_hour < wake_hour):  # Sleep after midnight: e.g. 1:00 to 9:00 or 14:00 to 17:00
        return (wake_hour - bed_hour) + (wake_minute - bed_minute)/60
    else:  # E.g. 20:15 to 4:30 or 10:00 to 8:00
        # Calculate the number of hours until midnight
        hr_to_midnight = (23 - bed_hour) + (60 - bed_minute)/60
        hr_after_midnight = wake_hour + (wake_minute/60)
        return (hr_to_midnight + hr_after_midnight)

def process_row(row):
    # Extract hours and minutes
    bed_hour, bed_minute = extract_time_components(row['psqi_01'])
    wake_hour, wake_minute = extract_time_components(row['psqi_03'])

    # Convert PSQI_02 and PSQI_04 to numeric, replacing non-numeric values with NaN
    time_fall_asleep = pd.to_numeric(row['psqi_02'], errors='coerce')
    rprt_sleep_dur = pd.to_numeric(row['psqi_04'], errors='coerce')

    # Check if any values are NaN
    if np.isnan(bed_hour) or np.isnan(bed_minute) or np.isnan(wake_hour) or np.isnan(wake_minute):
        return pd.Series({
            'bed_hour': np.nan,
            'bed_minute': np.nan,
            'wake_hour': np.nan,
            'wake_minute': np.nan,
            'time_in_bed': np.nan,
            'time_in_bed_m12': np.nan,
            'time_fall_asleep': time_fall_asleep,
            'est_sleep_dur': np.nan,
            'est_sleep_dur_m12': np.nan,
            'rprt_sleep_dur': rprt_sleep_dur
        })
    else:
        t = hrs_in_bed_time(bed_hour, bed_minute, wake_hour, wake_minute)
        if (t > 15):
            t2 = t - 12  # Accounting for possibly inputing the time to bed in 12- instead of 24-hr format
        else:
            t2 = t

        # Calculate est_sleep_dur only if time_fall_asleep is not NaN
        est_sleep_dur = np.nan if np.isnan(time_fall_asleep) else round(t, 2) - time_fall_asleep/60
        est_sleep_dur_m12 = np.nan if np.isnan(time_fall_asleep) else round(t2, 2) - time_fall_asleep/60

        # Return new df
        return pd.Series({
            'bed_hour': round(bed_hour),
            'bed_minute': round(bed_minute),
            'wake_hour': round(wake_hour),
            'wake_minute': round(wake_minute),
            'time_in_bed': round(t, 2),
            'time_in_bed_m12': round(t2, 2),
            'time_fall_asleep': time_fall_asleep,
            'est_sleep_dur': est_sleep_dur,
            'est_sleep_dur_m12': est_sleep_dur_m12,
            'rprt_sleep_dur': rprt_sleep_dur
        })

columns = ['bed_hour', 'bed_minute', 'wake_hour', 'wake_minute',
           'time_in_bed', 'time_in_bed_m12', 'time_fall_asleep',
           'est_sleep_dur', 'est_sleep_dur_m12', 'rprt_sleep_dur']

df_PSQI[columns] = df_PSQI.apply(process_row, axis=1)
print(df_PSQI.columns.to_list())


# Calculate Total_Sleep_Score
# https://www.sleep.pitt.edu/sites/default/files/assets/Instrument%20Materials/Exhibit%20A-PSQI%20scoring.pdf

def calculate_psqi_score(row):
    # Component 1: Sleep Quality (PSQISLPQUAL)
    c1 = row['psqi_06']

    # Component 2: Sleep Latency (PSQILATEN)
    if pd.isna(row['time_fall_asleep']):
        q2_score = np.nan
    elif row['time_fall_asleep'] <= 15:
        q2_score = 0
    elif row['time_fall_asleep'] <= 30:
        q2_score = 1
    elif row['time_fall_asleep'] <= 60:
        q2_score = 2
    else:
        q2_score = 3

    c2_sum = row['psqi_05a'] + q2_score if pd.notna(row['psqi_05a']) and pd.notna(q2_score) else np.nan
    if pd.isna(c2_sum):
        c2 = np.nan
    elif c2_sum == 0:
        c2 = 0
    elif c2_sum < 3:  # >= 1 and <= 2
        c2 = 1
    elif c2_sum < 5:  # >= 3 and <= 4
        c2 = 2
    else:  # >= 5 and <= 6
        c2 = 3

    # Component 3: Sleep Duration (PSQIDURAT)
    if pd.isna(row['rprt_sleep_dur']):
        c3 = np.nan
    elif row['rprt_sleep_dur'] >= 7:
        c3 = 0
    elif row['rprt_sleep_dur'] >= 6:
        c3 = 1
    elif row['rprt_sleep_dur'] >= 5:
        c3 = 2
    else:
        c3 = 3

    # Component 4: Sleep Efficiency (PSQIHSE) using time_in_bed not mod 12
    if pd.isna(row['rprt_sleep_dur']) or pd.isna(row['time_in_bed']) or (row['time_in_bed'] == 0):
        c4 = np.nan
    else:
        sleep_efficiency = (row['rprt_sleep_dur'] / row['time_in_bed']) * 100
        if sleep_efficiency >= 85:
            c4 = 0
        elif sleep_efficiency >= 75:
            c4 = 1
        elif sleep_efficiency >= 65:
            c4 = 2
        else:
            c4 = 3

    # Component 5: Sleep Disturbances (PSQIDISTB)
    columns = ['psqi_05b', 'psqi_05c', 'psqi_05d', 'psqi_05e', 'psqi_05f',
               'psqi_05g', 'psqi_05h', 'psqi_05i', 'psqi_05j']

    # psqi_05j1 = Yes/No and psqi_05j2 is freq if either of these two is NA, set psqi_05j to 0
    # else if psqi_05j = 1 then take frequency from psqi_05j2
    # else NaN
    if pd.isnull(row['psqi_05j1']) or pd.isnull(row['psqi_05j2']):
        row['psqi_05j'] = 0
    elif row['psqi_05j1'] == 1:
        row['psqi_05j'] = row['psqi_05j2']
    else:
        row['psqi_05j'] = np.nan

    disturbances = sum(row[col] for col in columns if pd.notna(row[col]))
    if pd.isna(disturbances):
        c5 = np.nan
    elif disturbances == 0:
        c5 = 0
    elif disturbances <= 9:
        c5 = 1
    elif disturbances <= 18:
        c5 = 2
    else:
        c5 = 3

    # Component 6: Use of Sleep Medication (PSQIMEDS)
    c6 = row['psqi_07']

    # Component 7: Daytime Dysfunction (PSQIDAYDYS)
    c7sum = row['psqi_08'] + row['psqi_09'] if pd.notna(row['psqi_08']) and pd.notna(row['psqi_09']) else np.nan
    if pd.isna(c7sum):
        c7 = np.nan
    elif c7sum > 4:  # >= 5 and <= 6
        c7 = 3
    elif c7sum > 2:  # >= 3 and <= 4
        c7 = 2
    elif c7sum > 0:  # >= 1 and <= 2
        c7 = 1
    else:
        c7 = 0

    # Calculate total PSQI score
    components = [c1, c2, c3, c4, c5, c6, c7]
    if any(pd.isna(c) for c in components):
        total_score = np.nan
    else:
        total_score = sum(components)

    return total_score


# Check for NaNs
def is_valid_row(row, columns):
    return all(pd.notna(row[col]) for col in columns)

# If none of relevant columns is NaN, then calculate total score
numeric_columns = ['time_in_bed_m12', 'time_fall_asleep', 'rprt_sleep_dur', 'psqi_05b', 'psqi_05c', 'psqi_05d', 'psqi_05e', 'psqi_05f',
                   'psqi_05g', 'psqi_05h', 'psqi_05i', 'psqi_06', 'psqi_07', 'psqi_08', 'psqi_09']

for col in numeric_columns:
    df_PSQI[col] = pd.to_numeric(df_PSQI[col], errors='coerce')

df_PSQI['Total_Sleep_Score'] = df_PSQI.apply(
    lambda row: calculate_psqi_score(row) if is_valid_row(row, numeric_columns) else np.nan, axis=1)


# Plot histogram of Total_Sleep_Score
plt.figure(figsize=(10, 6))
sns.histplot(df_PSQI['Total_Sleep_Score'].dropna(), bins=20, kde=True, color='skyblue')
plt.title('Distribution of Total Sleep Score', fontsize=16, fontweight='bold')
plt.xlabel('Total Sleep Score', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


# Save the updated DataFrame to a new CSV file
# Edit save path as needed
output_file = 'PSQI_recalc.csv'
df_PSQI.to_csv(output_file, index=False)