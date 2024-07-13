# %% IMPORTING/READING
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.linear_model import LinearRegression
import missingno as mno
from sklearn.preprocessing import MinMaxScaler
from statsmodels.imputation.mice import MICEData
from IPython.display import display
from datetime import timedelta
import plotly.graph_objects as go
import streamlit as st
import sys
sys.path.append('LinkedListClass.py')
from LinkedListClass import Node, LL



# %% cleaning and setting data structures


#               ---------- BASIC CLEANING ------------


vitalsP_DF = pd.read_csv('vitalsP.csv')
vitalsP_DF = vitalsP_DF.drop(columns=['Unnamed: 0', 'observationoffset', 'Day', 'Hour', 'systemicdiastolic'])
vitalsP_DF['MeanBP'] = np.nan
vitalsP_DF['SystolicBP'] = np.nan
vitalsP_DF['HR'] = np.nan
vitalsP_DF['RR'] = np.nan
vitalsP_DF['Temp'] = np.nan
vitalsP_DF['Sats'] = np.nan


#               ---------- CREATING A MULTI-INDEX ----------


vitalsP_DF = vitalsP_DF.sort_values(by=['patientunitstayid', 'Time']) # sort id, then time within id
vitalsP_MultiIndex = vitalsP_DF.set_index(['patientunitstayid', 'Time'])
# check if multiindex worked
display(vitalsP_MultiIndex)


#            ---------- CLEANING PATEINT ID'S ----------


# list of all patient id's
total_patient_list =  [
    306989, 1130290, 1210520, 1555058, 1580984, 2375786, 2823473,
    2887235, 2898120, 3075566, 3336597, 193629, 263556, 272638, 621883, 799478, 1079428, 1082792,
    1088266, 1092809, 1116007, 1162658, 1175888, 1535342, 1556670, 2198292,
    2247037, 2405050, 2671145, 2683425, 2689775, 2721908, 2722053, 2724565,
    2725853, 2767039, 2768739, 2773734, 2803129, 2846229, 2870532,
    2885054, 2890935, 2895083, 3064120, 3100062, 3210988, 3212405, 3214569,
    3217832, 3222024, 3245093, 3347750, 2782239
]
# list of unique id's
vitalsP_DF_patientIDs = vitalsP_MultiIndex.index.get_level_values('patientunitstayid').unique()
# find missing id's from total list
orig_set, gen_set = set(total_patient_list), set(vitalsP_DF_patientIDs)
missing_in_vitalsP = orig_set - gen_set
print("patient 1082792 has only a single data point")
print(f"missing ids from og list but not in gen list {missing_in_vitalsP}")

# vitalsP_DF and vitalsP_MultiIndex now only retains the patient ids that are in patient_list
vitalsP_DF = vitalsP_DF.loc[vitalsP_DF['patientunitstayid'].isin(total_patient_list)]
vitalsP_MultiIndex = vitalsP_MultiIndex.loc[vitalsP_MultiIndex.index.
                     get_level_values('patientunitstayid').isin(total_patient_list)]


#            ---------- CREATING A LINKED LIST ----------


# create a linked list from vitalsP_MultiIndex
vitalsP_LL = LL()
# each pateint ID makes a mutli index object, of which is stored in the linked list 
for patient_id in vitalsP_DF_patientIDs:
    # creates a multii index object by patient id
    df_multiobj = vitalsP_MultiIndex[vitalsP_MultiIndex.index.get_level_values('patientunitstayid') == patient_id]
    # adds to LL
    vitalsP_LL.append(df_multiobj)
# check if linked list is working
vitalsP_LL.display()
print(vitalsP_LL.length())


#            ---------- SPLIT vitalsP INTO EXPIRED AND ALIVE GROUPS ----------


print("note that for patient 1210520, there is a discrepancy between the two values")
print("we sure that we're ok with that?")

# list of expired patient id's
expiredID_list = [306989, 1130290, 1210520, 1555058, 1580984, 2375786, 2823473,
    2887235, 2898120, 3075566, 3336597]
# list of alive patient id's
aliveID_list = [193629, 263556, 272638, 621883, 799478, 1079428, 1082792,
    1088266, 1092809, 1116007, 1162658, 1175888, 1535342, 1556670, 2198292,
    2247037, 2405050, 2671145, 2683425, 2689775, 2721908, 2722053, 2724565,
    2725853, 2767039, 2768739, 2773734, 2803129, 2846229, 2870532,
    2885054, 2890935, 2895083, 3064120, 3100062, 3210988, 3212405, 3214569,
    3217832, 3222024, 3245093, 3347750, 2782239]


# Expired and Alive patient id's
expiredID_DF = vitalsP_DF[vitalsP_DF['patientunitstayid'].isin(expiredID_list)]
aliveID_DF = vitalsP_DF[vitalsP_DF['patientunitstayid'].isin(aliveID_list)]


#               -------------- Convert LL to list (easy manipulation) --------------

vitalsP_nodeList = []
current_node = vitalsP_LL.head
while current_node:
    vitalsP_nodeList.append(current_node)
    current_node = current_node.next


#               -------------- Replace negative values --------------


# Iterate through the list using a for loop
for node in vitalsP_nodeList:
    node.data.loc[node.data['systemicmean'] < 0, 'systemicmean'] = np.nan
    node.data.loc[node.data['systemicsystolic'] < 0, 'systemicsystolic'] = np.nan
    node.data.loc[node.data['heartrate'] < 0, 'heartrate'] = np.nan
    node.data.loc[node.data['respiration'] < 0, 'respiration'] = np.nan
    node.data.loc[node.data['sao2'] < 0, 'sao2'] = np.nan
    node.data.loc[node.data['temperature'] < 25, 'temperature'] = np.nan
    print(node.data)

mno.matrix(vitalsP_DF, figsize=(20, 6)) # displays NaN's in df_vitalsP

# %%
def imputeClosest_systemicmean(df, window):
    # Ensure 'Time' column is in float format
    original_time = df['Time'].copy()

    # Convert 'Time' to datetime for imputation
    df = df.copy()  # Ensure we're working on a copy of the DataFrame to avoid SettingWithCopyWarning
    df.loc[:, 'Datetime'] = pd.to_datetime(df['Time'], unit='h')

    df = df[df['systemicmean'] >= 0]

    for i in df[df['systemicmean'].isna()].index:
        current_time = df.at[i, 'Datetime']
        time_window = pd.Timedelta(hours=window)

        mask = ((df['Datetime'] >= current_time - time_window) & (df['Datetime'] <= current_time + time_window))
        candidates = df.loc[mask & df['systemicmean'].notna(), 'systemicmean']

        if not candidates.empty:
            closest_index = (df.loc[candidates.index, 'Datetime'] - current_time).abs().idxmin()
            closest_value = df.at[closest_index, 'systemicmean']
            df.loc[i, 'systemicmean'] = closest_value

    # Revert 'Time' back to original float format
    df.loc[:, 'Time'] = original_time
    df = df.drop(columns='Datetime')
    
    return df


#%%
# Process each node in the linked list, impute missing values, and store the results in a new linked list
tempNode = vitalsP_LL.head
vitalsP_imputed_LL = LL()

while tempNode:
    dt = tempNode.data
    dt = dt.reset_index()
    if not isinstance(dt, pd.Series) and (len(dt) != 0):
        dt_imputed = imputeClosest_systemicmean(dt, 5)
        vitalsP_imputed_LL.append(dt_imputed)
    tempNode = tempNode.next

# Concatenate all dataframes from the imputed linked list into a single dataframe
tempNode = vitalsP_imputed_LL.head
vitalsP_imputed_DF = pd.DataFrame()

while tempNode:
    dt = tempNode.data
    vitalsP_imputed_DF = pd.concat([vitalsP_imputed_DF, dt], ignore_index=True)
    tempNode = tempNode.next

# Display the final concatenated dataframe

vitalsP_imputed_DF.head(1000000000000000000000000000000000000000000000000)
# %%
mno.matrix(vitalsP_imputed_DF, figsize=(20, 6)) 


# %%

