# %% intro
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import missingno as msno  # Import missingno library
from LinkedListClass import Node, LL
import re
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

import plotly.express as px


# lets use seaborn for the sake of optimization here. 
import seaborn as sns 

df = pd.read_csv('infsP_results.csv')
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns


# There's so few, just drop them for now. 
# Remove any rows with NaN values
df = df.dropna()



# Drop columns
df = df.drop(columns=['Unnamed: 0', 'infusionoffset'])
# convert object type to float
print("Data types of each column:\n", df.dtypes)
df['drugrate'] = pd.to_numeric(df['drugrate'], errors='coerce')
print("Data types of each column:\n", df.dtypes)
rowList = df[df['drugrate'].isnull()].index.tolist()
# nan graph and nan rows
msno.matrix(df)
plt.show()
print(len(rowList))
print(rowList)
# drop rows with nan values
df = df.dropna()
# normalize drugrate for each drug
scaler = MinMaxScaler()

drugList = df['drugname'].unique()
for drug in drugList:

    drug_df = df[df['drugname'] == drug].copy()
    drug_df['drugrate'] = scaler.fit_transform(drug_df[['drugrate']])
    df.loc[df['drugname'] == drug, 'drugrate'] = drug_df['drugrate']
    print(df[df['drugname'] == drug])



# %%

# Assuming df has a column 'patient_id' for unique patient identifiers
unique_patients = df['patientunitstayid'].unique()

# Create a figure with subplots for each patient
n_patients = len(unique_patients)
ncols = 3  # Number of columns in the grid
nrows = (n_patients // ncols) + (n_patients % ncols > 0)  # Number of rows needed

# Increase the width of the figure
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(40, 5 * nrows), sharex=True, sharey=True)

# Flatten axes array for easy iteration
axes = axes.flatten()

for i, patient_id in enumerate(unique_patients):
    ax = axes[i]
    
    # Filter data for the current patient
    patient_df = df[df['patientunitstayid'] == patient_id].copy()  # Use .copy() to avoid warnings

    # Create the scatterplot for the current patient
    sns.scatterplot(data=patient_df, x='Time', y='drugrate', hue='drugname', palette='Set2', s= 100, alpha=0.5, edgecolor=None, ax=ax)

    # Set titles and labels
    ax.set_title(f'Patient {patient_id}', fontsize=14)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Normalized Drug Rate', fontsize=12)


    # # LINE PLOT EXAMPLE
    # # Create a line plot for the current patient
    # sns.lineplot(data=patient_df, x='Time', y='drugrate', hue='drugname', palette='Set2', ax=ax, markers=True)
    
    # # Set titles and labels
    # ax.set_title(f'Patient {patient_id}', fontsize=14)
    # ax.set_xlabel('Time', fontsize=12)
    # ax.set_ylabel('Normalized Drug Rate', fontsize=12)


    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='Drug Name', bbox_to_anchor=(1.05, 1), loc='upper left')

# Remove any empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Adjust layout to make room for legends
plt.tight_layout(rect=[0, 0, 0.8, 1])  # Adjust the right side to make room for the legend
plt.show()

# %%

# Load and preprocess the data
df = pd.read_csv('infsP_results.csv')
df = df.dropna()
df = df.drop(columns=['Unnamed: 0', 'infusionoffset'])
df['drugrate'] = pd.to_numeric(df['drugrate'], errors='coerce')
df = df.dropna()

# Normalize drugrate for each drug
scaler = MinMaxScaler()
drugList = df['drugname'].unique()
for drug in drugList:
    drug_df = df[df['drugname'] == drug].copy()
    drug_df['drugrate'] = scaler.fit_transform(drug_df[['drugrate']])
    df.loc[df['drugname'] == drug, 'drugrate'] = drug_df['drugrate']

# Unique patient IDs
unique_patients = df['patientunitstayid'].unique()

# Create a figure with subplots for each patient
n_patients = len(unique_patients)
ncols = 3
nrows = (n_patients // ncols) + (n_patients % ncols > 0)

# Increase the width of the figure
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(40, 5 * nrows), sharex=True, sharey=True)

# Flatten axes array for easy iteration
axes = axes.flatten()

for i, patient_id in enumerate(unique_patients):
    ax = axes[i]
    
    # Filter data for the current patient
    patient_df = df[df['patientunitstayid'] == patient_id].copy()  # Use .copy() to avoid warnings

    # Create the line plot for the current patient
    sns.lineplot(data=patient_df, x='Time', y='drugrate', hue='drugname', palette='Set2', ax=ax, markers=True)

    # Set titles and labels
    ax.set_title(f'Patient {patient_id}', fontsize=14)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Normalized Drug Rate', fontsize=12)

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='Drug Name', bbox_to_anchor=(1.05, 1), loc='upper left')

# Remove any empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Adjust layout to make room for legends
plt.tight_layout(rect=[0, 0, 0.8, 1])  # Adjust the right side to make room for the legend
plt.show()

# %%
