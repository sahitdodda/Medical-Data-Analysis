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


# Assuming vitalsP_DF is already defined and reset_index() is applied

def imputeClosest_systemicmean(df, timeCol, icpCol, window):
    # Ensure 'Time' column is datetime
    df = df[df[icpCol] >= 0]

    df[timeCol] = pd.to_datetime(df[timeCol])

    for i in df[df[icpCol].isna()].index:
        current_time = df.at[i, timeCol]
        time_window = pd.Timedelta(hours=window)

        mask = ((df[timeCol] >= current_time - time_window) & (df[timeCol] <= current_time + time_window))
        candidates = df.loc[mask & df[icpCol].notna(), icpCol]

        if not candidates.empty:
            closest_index = (df.loc[candidates.index, timeCol] - current_time).abs().idxmin()
            closest_value = df.at[closest_index, icpCol]
            df.at[i, icpCol] = closest_value
    return df

# Correct usage: Pass column names as strings
imputedcrap = imputeClosest_systemicmean(vitalsP_DF, 'Time', 'systemicmean', 5)
print(imputedcrap)
#check status stuff

#%%
imputedcrap.head()
# %% Imputation
count = 0 

# noticed that for the current column, I want the systemic metric for that time
list.append(dt['systemticMetric'].iloc[count])



count += 1


#               -------------- Set variables --------------


fwd = 0 # time @ next systemicmean value
bwd = 0 # time # prev systemicmean value
fwdDist = 0 # dist. from current point to fwd time
bwdDist = 0 # dist. from current point to bwd time
nanRemove = 0 # total # of nan's removed
booleanBwd = True # whether or not to check if we backwards
booleanFwd = True # whether or not to check if we forwards

node = vitalsP_nodeList[0].data # orig dataframe
dt = node # copy of node where the imputed values will be entered

time = node.index.get_level_values('Time') # list of systemicmean times
timeMissingSysMean = time[node['systemicmean'].isna()] # list of times where systemicmean is missing


# note that only this is being printed 
print(node.to_string()) # print orig dataframe
print("systemicmean left", dt['systemicmean'].isna().sum()) # num. of nan's left in systemicmean


#              -------------- Imputation --------------


# iterate through each the missing times with variable 'spot' -> look fwd/bwd -> impute based on conditions
for spot in timeMissingSysMean:
            # reset variables
                booleanBwd = True
                booleanFwd = True
            # if empty data frame, skip imputation
                if(time.size == 0 or timeMissingSysMean.size == 0):
                    print("nothing to impute")
                    continue
                # time index of 'spot' to impute missing value
                spot_ind = np.where(time == spot)[0][0]
                fwd, bwd = time[spot_ind], time[spot_ind]

                # Check forward imputation possibility
                if spot_ind + 1 >= len(time):
                    booleanFwd = False
                else:
                    i = 0
                    # keep going fwd until a systemicmean value is found
                    while np.isnan(node.loc[(node.index.get_level_values('Time') == fwd), 'systemicmean'].values):
                        i+= 1
                        if spot_ind + i >= len(time):
                            booleanFwd = False
                            break
                        fwd = time[spot_ind + i] # save the time the systemicmean value was found
                        fwdDist = fwd - time[spot_ind] # dist from spot to fwd
                        # if the dist is greater than 5 hrs, don't impute
                        if(fwdDist > 5):
                            booleanFwd = False
                # Check backward imputation possibility
                if spot_ind - 1 < 0:
                    booleanBwd = False
                else:
                    i = 0
                    # keep going bwd until a systemicmean value is found
                    while np.isnan(node.loc[(node.index.get_level_values('Time') == bwd), 'systemicmean'].values):
                        i-= 1
                        if spot_ind - i < 0:
                            booleanFwd = False
                            break
                        bwd = time[spot_ind + i] # save the time the systemicmean value was found
                        bwdDist = time[spot_ind] - bwd # dist from spot to fwd
                        # if the dist is greater than 5 hrs, don't impute
                        if(bwdDist > 5):
                            booleanBwd = False


                # conditions to impute
                if not booleanBwd and not booleanFwd: # if both fwd and bwd are not possible, don't impute
                    print("can't impute")
                elif booleanFwd and not booleanBwd: # if only fwd is possible, impute with fwd
                    MeanBP_val = node.loc[(node.index.get_level_values('Time') == fwd), 'systemicmean'].values[0] # imputed value
                    print(f"spot: {spot}, fwd: {fwd}, MeanBP_val: {MeanBP_val}") # testing code by outputting values

                    dt.loc[(dt.index.get_level_values('Time') == spot), 'MeanBP'].values[0] = MeanBP_val # adding to dataframe
                    print("replace", dt.loc[(dt.index.get_level_values('Time') == spot), 'MeanBP'].values[0]) # testing if the dataframe had the new values

                    nanRemove += 1 # increment nanRemove counter
                elif not booleanFwd and booleanBwd: # if only bwd is possible, impute with bwd
                    MeanBP_val = node.loc[(node.index.get_level_values('Time') == bwd), 'systemicmean'].values[0] # imputed value
                    print(f"spot: {spot}, bwd: {bwd}, MeanBP_val: {MeanBP_val}") # testing code by outputting values

                    dt.loc[(dt.index.get_level_values('Time') == spot), 'MeanBP'].values[0] = MeanBP_val # adding to dataframe
                    print("replace", dt.loc[(dt.index.get_level_values('Time') == spot), 'MeanBP'].values[0]) # testing if the dataframe had the new values

                    nanRemove += 1 # increment nanRemove counter
                elif fwdDist < bwdDist: # if fwd is closer than bwd, impute with fwd
                    MeanBP_val = node.loc[(node.index.get_level_values('Time') == fwd), 'systemicmean'].values[0] # imputed value
                    print(f"spot: {spot}, fwd: {fwd}, MeanBP_val: {MeanBP_val}") # testing code by outputting values

                    dt.loc[(dt.index.get_level_values('Time') == spot), 'MeanBP'].values[0] = MeanBP_val # adding to dataframe
                    print("replace", dt.loc[(dt.index.get_level_values('Time') == spot), 'MeanBP'].values[0]) # testing if the dataframe had the new values

                    nanRemove += 1 # increment nanRemove counter
                elif fwdDist > bwdDist: # if bwd is closer than fwd, impute with bwd
                    MeanBP_val = node.loc[(node.index.get_level_values('Time') == bwd), 'systemicmean'].values[0] # imputed value
                    print(f"spot: {spot}, bwd: {bwd}, MeanBP_val: {MeanBP_val}") # testing code by outputting values

                    dt.loc[(dt.index.get_level_values('Time') == spot), 'MeanBP'].values[0] = MeanBP_val # adding to dataframe
                    print("replace", dt.loc[(dt.index.get_level_values('Time') == spot), 'MeanBP'].values[0]) # testing if the dataframe had the new values

                    nanRemove += 1 # increment nanRemove counter
                elif fwdDist == bwdDist: # if both are equidistant, impute with fwd
                    MeanBP_val = node.loc[(node.index.get_level_values('Time') == fwd), 'systemicmean'].values[0] # imputed value
                    print(f"spot: {spot}, fwd: {fwd}, MeanBP_val: {MeanBP_val}") # testing code by outputting values

                    dt.loc[(dt.index.get_level_values('Time') == spot), 'MeanBP'].values[0] = MeanBP_val # adding to dataframe
                    print("replace", dt.loc[(dt.index.get_level_values('Time') == spot), 'MeanBP'].values[0]) # testing if the dataframe had the new values

                    nanRemove += 1 # increment nanRemove counter
                

print(nanRemove) # prints nanRemove counter
print(dt.to_string()) # prints the updated dataframe
nanRemove = 0 # resets nanRemoval

