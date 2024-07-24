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

from sklearn.impute import KNNImputer


# %% cleaning and setting data structures


#               ---------- BASIC CLEANING ------------

vitalsP_DF = pd.read_csv('vitalsP.csv')
vitalsP_DF = vitalsP_DF.drop(columns=['Unnamed: 0', 'observationoffset', 'Day', 'Hour', 'systemicdiastolic'])
vitalsP_DF = vitalsP_DF[vitalsP_DF['patientunitstayid'] != 1082792]
# within 7 days (168 hr)
vitalsP_DF = vitalsP_DF[vitalsP_DF['Time'] <= 168]

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

# HUGE MENTION, DROPPING PATIENT WITH SINGLE DATA POINT 
vitalsP_DF = vitalsP_DF[vitalsP_DF['patientunitstayid'] != 1082792]

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

lenOrig = len(vitalsP_DF)

# %%
def imputeClosest(df, col, window):
    # Ensure 'Time' column is in float format
    original_time = df['Time'].copy()

    # Convert 'Time' to datetime for imputation
    df = df.copy()  # Ensure we're working on a copy of the DataFrame to avoid SettingWithCopyWarning
    df.loc[:, 'Datetime'] = pd.to_datetime(df['Time'], unit='h')

    # df = df[df[col] >= 0]

    for i in df[df[col].isna()].index:
        current_time = df.at[i, 'Datetime']
        time_window = pd.Timedelta(hours=window)

        mask = ((df['Datetime'] >= current_time - time_window) & (df['Datetime'] <= current_time + time_window))
        candidates = df.loc[mask & df[col].notna(), col]

        if not candidates.empty:
            closest_index = (df.loc[candidates.index, 'Datetime'] - current_time).abs().idxmin()
            closest_value = df.at[closest_index, col]
            df.loc[i, col] = closest_value

    # Revert 'Time' back to original float format
    df.loc[:, 'Time'] = original_time
    df = df.drop(columns='Datetime')
    
    return df

#%%
# Process each node in the linked list, impute missing values, and store the results in a new linked list
tempNode = vitalsP_LL.head
vitalsP_imputed_LL = LL()

colImpute = ['Time', 'temperature', 'sao2', 'heartrate', 'respiration', 'systemicsystolic', 'systemicmean']

while tempNode:
    dt = tempNode.data
    dt = dt.reset_index()
    dt = dt.sort_values(by='Time')
    if not isinstance(dt, pd.Series) and (len(dt) != 0):
        for col in colImpute:
            dt_imputed = imputeClosest(dt, col, 1000)
            dt = dt_imputed.copy()
        vitalsP_imputed_LL.append(dt)
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

no_temp_list = []
no_respiration_list = []
no_sys_list = []
no_mean_list = []

# test print
tempNode = vitalsP_imputed_LL.head
count = 1
while tempNode: 
    dt = tempNode.data
    dt = dt.reset_index()
    patient = dt['patientunitstayid']
    if(dt['temperature'].isnull().sum() == len(dt)):
        no_temp_list.append(patient)
    if(dt['respiration'].isnull().sum() == len(dt)):
        no_respiration_list.append(patient)
    if(dt['systemicsystolic'].isnull().sum() == len(dt)):
        no_sys_list.append(patient)
    if(dt['systemicmean'].isnull().sum() == len(dt)):
        no_mean_list.append(patient)
    print(mno.matrix(dt, figsize = (20,6)))
    count += 1
    tempNode = tempNode.next    
    
print(count)
# %%

print(f'No temp length {len(no_temp_list)}')
print(f'No respiration length {no_respiration_list}')
print(f'No systemicsystolic length {no_sys_list}')
print(f'No systemicmean length {no_mean_list}')

# %%

# we can use KNNs later for better inputation (after further discussion)

# For the sake of simplicity and moving forward, lets apply inputation again on 
    # respiration, systemicsystolic, and systemicmean
list = ['respiration', 'systemicsystolic', 'systemicmean']
# in context of the entire dataframe. 

# We can also drop temperature for now. 

vitalsP_imputed_DF = vitalsP_imputed_DF.drop(columns='temperature')

vitalsP_imputed_DF_copy = vitalsP_imputed_DF.copy()

# now call the function 

# for i in range(len(list)):
#     vitalsP_imputed_DF_copy = imputeClosest(vitalsP_imputed_DF_copy, list[i], 5)

imputer = KNNImputer(n_neighbors=5)

vitalsP_imputed_DF_copy = pd.DataFrame(imputer.fit_transform(vitalsP_imputed_DF_copy), columns=vitalsP_imputed_DF_copy.columns)

mno.matrix(vitalsP_imputed_DF_copy, figsize=(20, 6)) 

# %%
'''
   .-.
  /'v'\    - THE REST OF THE CODE IS HANDLING AUC FOR DAYS AND LEVELS.      
 (/   \)
='="="===< 
mrf|_|
'''

# %%

time_ranges = [(0, 24), (24, 48), (48, 72), (72, 96), (96, 120), (120, 144), (144, 168)]

def day_icp_load(patient_df, patient):
    df_time_ranges = LL()
    df_icp_loads = LL()
    for df_value in time_ranges: 

        min_time = df_value[0]
        max_time = df_value[1]  
        
        # Filter corresponding time column based on the same condition
        df_day = patient_df.loc[(patient_df['Time'] >= min_time) & (patient_df['Time'] <= max_time), ['icp', 'Time']]     

        df_time_ranges.append(df_day)
    
    # now use df_time_ranges

    icp_load = 0

    tempNode = df_time_ranges.head
    while tempNode:
        dt = tempNode.data
        dt = dt.sort_values(by='Time')
        icp_load = np.trapz(dt['icp'], dt['Time'])
        # append to the actual linked list 
        df_icp_loads.append(icp_load)
        tempNode = tempNode.next

    return df_icp_loads


# now traverse through linked list of patients and calcualte linked list of icp loads for each

patient_list = []
Time_list = []
# currently 7 days 
day_list = [[], [], [], [], [], [], [], []]


tempNode = vitalsP_imputed_LL.head

while tempNode: 
    dt = tempNode.data
    dt = dt.reset_index()
    patient_list.append(dt['patientunitstayid'].iloc[0])
    time = dt['Time']

    # icp load list, then iterate through the linked list, adding each as its own column
    icp_load_list = LL()
    icp_load_list = day_icp_load(dt, patient)

    tempNode_icp = icp_load_list.head
    count = 0

    while tempNode_icp:
        day_list[count].append(tempNode_icp.data) 
        tempNode_icp = tempNode_icp.next
        count += 1

    sum_area = np.trapz(dt['icp'], dt['Time'])
    day_list[7].append(sum_area)

    tempNode = tempNode.next

data = {
    'patientunitstayid' : patient_list, 
    'Day 1' : day_list[0],
    'Day 2' : day_list[1],
    'Day 3' : day_list[2],
    'Day 4' : day_list[3],
    'Day 5' : day_list[4],
    'Day 6' : day_list[5],
    'Day 7' : day_list[6], 
    'Summation' : day_list[7]
}

df_range = pd.DataFrame(data)
df_range.head(100000000000000000000000000000000000000)


#%%

# Pseuodocode 

# create two time columns: 
    # hour time column to help split by day. 
    # minute time column to help split by hour within each day
# 

# for now, we worry about hours in the context of 3 days.
# assume hour threshold is working with time in minutes, thus: 
# hour_threshold = []


# generic code for thresholds is here:

icp_threshold = [20, 25, 30, 35]
t_cond_icp = [False for _ in range(len(icp_threshold))]

Day_threshold = [24, 48, 72, 96, 120, 144, 168]
t_cond_Day = [False for _ in range(len(Day_threshold))]

# this now gives us hour thresholds for the first 3 days 
hour_threshold = [hour for hour in range(1,168)]
t_cond_hour = [False for _ in range(len(hour_threshold))]

all_thresholds = [
    (('icp', 'Time'), icp_threshold), 
    (('Time', 'icp'), Day_threshold)
]

def calc_thresh_val(y1, y2, x1, x2, threshold): # chose refers to icp or Time 
    ydiff = y2-y1
    xdiff = x2-x1
    if xdiff == 0:
        return y1, x1    
    slope = ydiff/xdiff

    new_value = (threshold-y1)/slope + x1
    return new_value, threshold

# for major ICP
calculated, threshold_value = calc_thresh_val(19, 31, 2, 4, 20)
print(f'CALCULATED imputation: {calculated}')
print(f'threshold used! {threshold_value}')

print('------------now using time as the major value ----------------')

# for major Time
calculated, threshold_value = calc_thresh_val(20, 25, 19, 31, 24)
print(f'CALCULATED imputation: {calculated}')
print(f'threshold used! {threshold_value}')

# %%

def find_thresholds(patient_df, thresholds, columnMajor, columnMinor, patient_list, l1, l2):
    previous_value = None
    previous_minor = None
    for index, row in patient_df.iterrows():
        current_value = row[columnMajor]
        current_minor = row[columnMinor]
        
        if previous_value is not None:
            crossed_thresholds = []
            for threshold in thresholds:
                if threshold is None:
                    continue
                if (previous_value < threshold <= current_value) or (previous_value > threshold >= current_value):
                    crossed_thresholds.append(threshold)
            
            # Sort crossed_thresholds based on their distance from previous_value
            crossed_thresholds.sort(key=lambda x: abs(x - previous_value))
            
            # Add crossed thresholds in the correct order
            for threshold in crossed_thresholds:
                calculated, threshold_value = calc_thresh_val(
                    previous_value, current_value,
                    previous_minor, current_minor,
                    threshold
                )
                patient_list.append(row['patientunitstayid'])
                l1.append(threshold_value)
                l2.append(calculated)
        
        # Always add the current value
        patient_list.append(row['patientunitstayid'])
        l1.append(current_value)
        l2.append(current_minor)
        
        previous_value = current_value
        previous_minor = current_minor
    
    return patient_list, l1, l2

data_sample = {
    'patientunitstayid' : [1, 1],
    'icp' : [19, 21],
    'Time' : [22.1, 26.5]
}
data_sample = pd.DataFrame(data_sample)
#%%
# debugging print for proof of concept. 

a = []
b = []
c = []
# patient_list, icp_list, time_list = find_thresholds(data_sample, icp_threshold, 'icp', 'Time', a, b, c)
# print("time list 1:")
# print(time_list)
# print("icp list 1")
# print(icp_list)

# # print('-----------------------------------------------------------')

# IMPORTANT NOTE: IT ONLY RETAINS ORDER IF YOU KEEP EVERYTHING IN THE SAME ORDER 
# BUT CAHNGE THE THRESHOLD AND THE MAJOR/MINOR VALUES. MAY NOT BE CONSISTENT FOR 3 ITERATIONS, BUT IS 
# CONSISTENT FOR 2 ITERATIONS. 

# IF YOU WANT TO DO IT ONLY ONE TIME, THEN THE ORDER DOES MATTER. IF YOU WANT TO DO IT TWICE, THE ORDER 
# NO LONGER MATTERS AS LONG AS IT IS CONSISTENT. 

# patient_list, icp_list, time_list  = find_thresholds(data_sample, icp_threshold, 'icp', 'Time',  a, b, c)
# print("time list 2:")
# print(time_list)
# print("icp list 2")
# print(icp_list)
# print(f'patient {patient_list}')
# print('----------------')

patient_list, time_list, icp_list  = find_thresholds(data_sample, Day_threshold, 'Time', 'icp', a, b, c)
print("time list 2:")
print(time_list)
print("icp list 2")
print(icp_list)
print(f'patient {patient_list}')

#%%
def ll_traversal(ll_list): # major indicates the type bc of the convention we are using. 
    # runs through icp and time twice. 

    temp_df_list = []
    final_df_list = []
    for patient_id in ll_list:
        patient_list = []
        icp_list = []
        time_list = []
        # major is icp and minor is Time, therefore the order is icp_list and time_list 
        patient_list, icp_list, time_list = find_thresholds(patient_id, icp_threshold, 'icp', 'Time', patient_list, icp_list, time_list)
        # patient_list, icp_list, time_list = find_thresholds(patient_id, Day_threshold, 'Time',  'icp', patient_list, icp_list, time_list)
        data = {
            'patientunitstayid' : patient_list, 
            'icp' : icp_list, 
            'Time' : time_list
        }
        temp_DF = pd.DataFrame(data)
        # print(temp_DF.head())
        temp_df_list.append(temp_DF)

    # do another for loop yayyyyyyy
    for patient_id in temp_df_list:
        patient_list = []
        icp_list = []
        time_list = []
        # major is Time and minor is icp, therefore the order is time_list and icp_list 
        patient_list, time_list, icp_list = find_thresholds(patient_id, hour_threshold, 'Time', 'icp', patient_list, icp_list, time_list)
        data = {
            'patientunitstayid' : patient_list, 
            'icp' : icp_list, 
            'Time' : time_list
        }
        temp_DF = pd.DataFrame(data)
        print(temp_DF.head()) # debug statement to make sure things don't swap incorrectly. 
        final_df_list.append(temp_DF)
    
    return final_df_list



# creates a list of a list of each patient's icp points
plotPointsNew_list = []
for patient_id in vitalsP_DF_patientIDs: 
    # holds original set of points
    if vitalsP_imputed_DF_copy.loc[vitalsP_imputed_DF_copy['patientunitstayid'] == patient_id, ['patientunitstayid', 'Time', 'icp']].empty:
        continue
    plotPoints = vitalsP_imputed_DF_copy.loc[vitalsP_imputed_DF_copy['patientunitstayid'] == patient_id, ['patientunitstayid', 'Time', 'icp']]
    plotPointsNew_list.append(plotPoints)

plotPointsNew_List = ll_traversal(plotPointsNew_list)

 
# %%

# -------------------------------------- Pseudocode --------------------------------------

# all patients (node in vitalsP_LL)
    # each patient (node.data (vitalsP for that ID))
        # each hour (create 168 huurs from node.data)
            # each point (find new points, spike count, spike times, spike durations)

                # save new points, spike count, spike times, spike durations into a dataframe
                # save dataframe into a totalDay_List (hour --167)
                # reset variables for next day
            # save totalDay_list into a totalPatient_list (in same order as vitalsP_LL)


# thresholds
thresholds = [20, 25, 30, 35]
t20 = False
t25 = False
t30 = False
t35 = False
t_cond = [t20, t25, t30, t35]


# all patients (node in vitalsP_LL)
    # each patient (node.data (vitalsP for that ID))
        # each day (create 7 days from node.data)
            # each point (find new points, spike count, spike times, spike durations)

                # save new points, spike count, spike times into a list
                # save dataframe into a totalDay_List (day 1-7)
                # reset variables for next day
            # save totalDay_list into a totalPatient_list (in same order as vitalsP_LL)


# all patients (node in vitalsP_LL)
    # each patient (node.data (vitalsP for that ID))
        # each day (create 7 days from node.data)
            # each hour (create 24 hours from node.data)
                # each point (find new points, spike count, spike times, spike durations)

                    # save new points, spike count, spike times into a list
                    # save dataframe into a totalHour_List (hour 1-168)
                    # reset variables for next hour
                    
                # save totalHur_list into a totalPatient_list (in same order as vitalsP_LL)


spikeStats_All = []
newGraph_All = []


node = vitalsP_imputed_LL.head
numNode = 0
while node: # by patient
    print(f'Node {numNode}')
    df = node.data # datafarame per ID 
    ID = df['patientunitstayid'].iloc[0] # patient ID

    # DEBUG PRINT STATEMENT
    print(ID)

    # saves OG points (hour 0-167) into a list
    HP_List = [df.loc[(df['Time'] >= i) & (df['Time'] < i + 1)] for i in range(168)]

    # reset day for next patient
    hour = 0
    newGraph_Patient = []
    spikeStats_Patient = []

    lastDay = 0
    lastHour = 0
    first = False

    for graph in HP_List: # each day

        hour += 1

        newGraph = pd.DataFrame(columns=['patientunitstayid', 'Time', 'icp']) # holds points
        spikeStats = pd.DataFrame(columns=['patientunitstayid', 'lastDay', 'spikes', 'spikeStarts', 'spikeEnds']) # holds lists
        
        spike_Count20, spike_Count25, spike_Count30, spike_Count35 = 0, 0, 0, 0
        start20, start25, start30, start35 = [], [], [], []
        end20, end25, end30, end35 = [], [], [], []
        
        start = [start20, start25, start30, start35]
        spikes = [spike_Count20, spike_Count25, spike_Count30, spike_Count35]
        end = [end20, end25, end30, end35]

        for i in range(len(graph)-1): # each point in graph
            
            # sets current and next point (used for conditions)
            nT = graph['Time'].iloc[i]
            nI = graph['icp'].iloc[i]
            nxT = graph['Time'].iloc[i+1]
            nxI = graph['icp'].iloc[i+1]
            
            # append the current point to graph
            newRow = pd.DataFrame({'patientunitstayid': ID, 'Hour': hour, 'Time': nT, 'icp': nI}, index=[0])
            # Check if newRow is not empty or all-NA before concatenation
            if not newRow.isna().all(axis=None):
                newGraph = pd.concat([newGraph, newRow], ignore_index=True)


            # sets threshold conditions
            if(nI == nxI): # skips if equal icp's
                continue
            for i in range(len(thresholds)): # when threshold is crossed, set condition
                if((nI < thresholds[i] and thresholds[i] < nxI) or (nI > thresholds[i] and thresholds[i] > nxI)):
                    t_cond[i] = True
            
            # finds slope
            slope = (nxI - nI) / (nxT - nT)

            # if passing threshold conditions
            # crosses 20
            if(t_cond[0]):
                x = ((20-nI)/slope) + nT
                
                # add new point to graph
                newRow = pd.DataFrame({'patientunitstayid': ID, 'Hour': hour, 'Time': x, 'icp': 20}, index=[0])
                if not newRow.isna().all(axis=None):
                    newGraph = pd.concat([newGraph, newRow], ignore_index=True)

                # starting/ending spikes
                if(slope > 0): # if slope is positive and goes thru threshold, it starts a spike
                    spike_Count20 += 1
                    start20.append(x)
                if(slope < 0): # if slope is negative and goes thru threshold, it ends a spike
                    end20.append(x)
            # crosses 25
            if(t_cond[1]):
                x = ((25-nI)/slope) + nT
                
                # add new point to graph
                newRow = pd.DataFrame({'patientunitstayid': ID, 'Hour': hour, 'Time': x, 'icp': 25}, index=[0])
                if not newRow.isna().all(axis=None):
                    newGraph = pd.concat([newGraph, newRow], ignore_index=True)

                if(slope > 0): # if slope is positive and goes thru threshold, it starts a spike
                    spike_Count25 += 1
                    start25.append(x)
                if(slope < 0): # if slope is negative and goes thru threshold, it ends a spike
                    end25.append(x)
            # crosses 30
            if(t_cond[2]):
                x = ((30-nI)/slope) + nT

                # add new point to graph
                newRow = pd.DataFrame({'patientunitstayid': ID, 'Hour': hour, 'Time': x, 'icp': 30}, index=[0])
                if not newRow.isna().all(axis=None):
                    newGraph = pd.concat([newGraph, newRow], ignore_index=True)
                
                if(slope > 0): # if slope is positive and goes thru threshold, it starts a spike
                    spike_Count30 += 1
                    start30.append(x)
                if(slope < 0): # if slope is negative and goes thru threshold, it ends a spike
                    end30.append(x)
            # crosses 35
            if(t_cond[3]):
                x = ((35-nI)/slope) + nT

                # add new point to graph
                newRow = pd.DataFrame({'patientunitstayid': ID, 'Hour': hour, 'Time': x, 'icp': 35}, index=[0])
                if not newRow.isna().all(axis=None):
                    newGraph = pd.concat([newGraph, newRow], ignore_index=True)

                if(slope > 0): # if slope is positive and goes thru threshold, it starts a spike
                    spike_Count35 += 1
                    start35.append(x)
                if(slope < 0): # if slope is negative and goes thru threshold, it ends a spike
                    end35.append(x)


            
            # if threshold is passed starting at threshold (runs when the other conditionals aren't ran. Ensured that we don't get too many values))
            
            if( (nI == 20 and nxI > 20) ): # start at 20, go 20+
                spike_Count20 += 1
                start20.append(x)
            if( (nI > 20 and nxI == 20) ): # start 20+, go to 20
                end20.append(x)
            
            if( (nI == 25 and nxI > 25) ): # start at 20, go 20+
                spike_Count25 += 1
                start25.append(x)
            if( (nI > 25 and nxI == 25) ): # start 20+, go to 20
                end25.append(x)
            
            if( (nI == 30 and nxI > 30) ): # start at 20, go 20+
                spike_Count30 += 1
                start30.append(x)
            if( (nI > 30 and nxI == 30) ): # start 20+, go to 20
                end30.append(x)
            
            if( (nI == 35 and nxI > 35) ): # start at 20, go 20+
                spike_Count35 += 1
                start35.append(x)
            if( (nI > 35 and nxI == 35) ): # start 20+, go to 20
                end35.append(x)

            # append the last point to graph
            if(i == len(graph)-2):
                newRow = pd.DataFrame({'patientunitstayid': ID, 'Hour': hour, 'Time': nxT, 'icp': nxI}, index=[0])
                # Check if newRow is not empty or all-NA before concatenation
                if not newRow.isna().all(axis=None):
                    newGraph = pd.concat([newGraph, newRow], ignore_index=True)


            # reset condiitons to prep for next point
            t_cond[0] = False
            t_cond[1] = False
            t_cond[2] = False
            t_cond[3] = False
    
        # append newGraph to total Patient list
        newGraph_Patient.append(newGraph)
        # save last day and hour for the patient
        if(newGraph.empty and first == False):
            lastHour = (df['Time'].iloc[-1])
            first = True

        # append statsList to total Patient list
        spikes = [spike_Count20, spike_Count25, spike_Count30, spike_Count35]
        start = [start20, start25, start30, start35]
        end = [end20, end25, end30, end35]

        spikeStats = pd.DataFrame({'patientunitstayid': ID, 'Hour': hour, 'spikes': spikes, 'lastDay': lastDay, 'lastHour': lastHour, 'spikeStarts': start, 'spikeEnds': end})
        spikeStats_Patient.append(spikeStats)


# Graph code to show later. 

# ------------------------------------------------- PRINTING VALUES BY DAY PER PATIENT -------------------------------------------------

        # print('spike count 20:', spike_Count20)
        # print('spike count 25:', spike_Count25)
        # print('spike count 30:', spike_Count30)
        # print('spike count 35:', spike_Count35)
        # print('start 20:', start20)
        # print('end 20:', end20)
        # print('start 25:', start25)
        # print('end 25:', end25)
        # print('start 30:', start30)
        # print('end 30:', end30)
        # print('start 35:', start35)
        # print('end 35:', end35)

# ------------------------------------------------- GRAPHING BY DAY PER PATIENT -------------------------------------------------

        # plt.figure(figsize=(10, 6))
        # # Plotting the graph
        # plt.plot(newGraph['Time'], newGraph['icp'], marker='o', linestyle='-')
        # # Adding horizontal lines at specific icp values
        # plt.axhline(y=20, color='r', linestyle='--', label='Threshold 20')
        # plt.axhline(y=25, color='g', linestyle='--', label='Threshold 25')
        # plt.axhline(y=30, color='b', linestyle='--', label='Threshold 30')
        # plt.axhline(y=35, color='y', linestyle='--', label='Threshold 35')
        # # Adding title and labels
        # plt.title(f'{ID}, Hour {hour}: ICP vs Time')
        # plt.xlabel('Time')
        # plt.ylabel('ICP')
        # # Adding legend
        # plt.legend()
        # # Display the plot
        # plt.show()

        # reset conditions for next day
        spike_Count20, spike_Count25, spike_Count30, spike_Count35 = 0, 0, 0, 0
        start20, start25, start30, start35 = [], [], [], []
        end20, end25, end30, end35 = [], [], [], []


    spikeStats_All.append(spikeStats_Patient)
    newGraph_All.append(newGraph_Patient)
    lastHour = 0
    first = False

    numNode += 1
    node = node.next


# %%
# we have plotPointsList and spikeStats_ALL 

# in spike stats all, we have what per patient list :
    # spike count, spike start time, spike end time, and for eahc of the 7 days. 

# index 0: first day 
# # index 1: 

# spikeStats_All (list that holds all 53 patient’s stats)

# spikeStats_Patient (list that has spike info for the 7 days in a patient)

# spikeStats (a single dataframe that has all the spike info)

# spikes (a list that has the spike counts from 20-35)

#  patient  | time period | total time icp was measured |  total auc for the time period | !!!spike 20 auc for time period !!!... | !!spike 20 num for time period!!  | 
#  19                                                                           

# with this, I want to make a dataframe of length 53:

# patient | time period | num spike 20 | num spike 25 | 
# 19369      day 1           9               2 
# 19369      day 2           8               3 

# one patient spike dataframe
patient_list = []
time_period = []
icp_period = []
num_spike_20 = []
num_spike_25 = []
num_spike_30 = []
num_spike_35 = []

hour = 1
sum_icp = 0


for list in spikeStats_All: # cycles through each patient
    
    for i in range(len(list)): # cycles through each day

        # patientID
        patientID = list[i]['patientunitstayid'].iloc[0]
        # time period
        hour = i+1

        # spike stats
        spike_Count20 = list[i]['spikes'][0]
        spike_Count25 = list[i]['spikes'][1]
        spike_Count30 = list[i]['spikes'][2]
        spike_Count35 = list[i]['spikes'][3]
        # append to the list
        patient_list.append(patientID)
        time_period.append(hour)
        num_spike_20.append(spike_Count20)
        num_spike_25.append(spike_Count25)
        num_spike_30.append(spike_Count30)
        num_spike_35.append(spike_Count35)

        
data = {
    'patientunitstayid' : patient_list, 
    'Hour (24-72 hours)' : time_period, 
    'num_spike_20' : num_spike_20, 
    'num_spike_25' : num_spike_25,
    'num_spike_30' : num_spike_30,
    'num_spike_35' : num_spike_35
}

megaStats_DF = pd.DataFrame(data)

# for debugging in the first 24 hours; is correct as we have 7 num20 spikes 
megaStats_DF.head(10000000000000000000)


 # %%

# #  -------------------- CALCULATING AUC, CORRECT VERSION FOR DAYS  ---------------------


# # spikeStats_All (holds the 53 patients)
#     # spikeStats_Patient (holds 168 hours)
#         # spikeStats (has the info for each hour)



# # tuple_day_threshold = [(0, 24), (24, 48), (48, 72), (72, 96), (96, 120), (120, 144), (144, 168)]
# tuple_hourly = [(i, i+1) for i in range(0, 168)]
# # tuple_hourly = [() for _ len()]
# patient_list2 = []
# hour2 = []
# total_AUC = []

# less_20_list = [] 

# # Function to shift ICP values
# def shift_icp_values(df, shift_amount):
#     df['icp'] = df['icp'] - shift_amount
#     df = df[df['icp'] >= 0]
#     return df

# def less20_area(df):
#     df = df.loc[df['icp'] <= 20, ['icp', 'Hour']]
#     return df

# # Function to calculate AUC for given thresholds
# def calc_auc(df):
#     results = []
#     for threshold in icp_threshold: 
#         data_above_threshold = df.loc[df['icp'] >= threshold, ['icp', 'Hour']]
#         data_above_threshold = shift_icp_values(data_above_threshold, threshold)
#         if not data_above_threshold.empty:
#             x = data_above_threshold['Hour'].values
#             y = data_above_threshold['icp'].values
#             area = np.trapz(y, x)
#             results.append(area)
#         else:
#             results.append(0)
#     return results

# # patient_list = []
# auc_list = [[] for _ in range(len(icp_threshold))]
# test_list = []

# for patient_df in plotPointsNew_List:
#     # Ensure df is not a Series and not empty
#     if not isinstance(patient_df, pd.Series) and (len(patient_df) != 0):
#         count = 0
#         patient_id = patient_df['patientunitstayid'].iloc[0].values
        
#         patient_df = patient_df.loc[patient_df['Hour'] >= 0]
#         patient_df = patient_df.sort_values(by=['Hour'])
#         for min_time, max_time in tuple_hourly:
#             day_df = patient_df.loc[(patient_df['Hour'] >= min_time) & (patient_df['Hour'] < max_time)]
            
#             # now that day df is working with the current day; 
#             if(day_df.empty):
#                 patient_list2.append(patient_id)
#                 # all values in auc_list[i] append 0 
#                 for i in range(len(auc_list)):
#                     auc_list[i].append(0)
#                 count += 1
#                 hour2.append(count)
#                 test_list.append(0)
#                 less_20_list.append(0)
#                 continue

#             # Append to the list         
#             patient_list2.append(patient_id)
#             auc_result = calc_auc(day_df)

#             for i in range(len(icp_threshold)):
#                 auc_list[i].append(auc_result[i])
#             test_list.append(np.trapz(day_df['icp'], day_df['Hour']))

#             test_area = less20_area(day_df)
#             less_20_list.append(np.trapz(test_area['icp'], test_area['Hour']))

#             count += 1
#             hour2.append(count)
        

# # Create the DataFrame

# data = {
#     'patientunitstayid': patient_list2, 
#     'Hour' : hour2,
#     '>20 auc': auc_list[0],
#     '>25 auc': auc_list[1],
#     '>30 auc': auc_list[2],
#     '>35 auc': auc_list[3], 
#     'Total (tested)': test_list,
#     '<20' : less_20_list
# }

# megaAUC_DF = pd.DataFrame(data)
# megaAUC_DF.head(100000000000000000000000000000)

# %%

# newGraph_All (holds the 53 patients)
    # newGraph_Patient (holds 168 dataframes for each hour)
        # newGraph (single dataframe for one hour)'

# newGraph_All (holds the 53 patients)
    # newGraph_Patient (holds 168 hours as a single dataframe)


newGraph_AllFinal = []
for patient in newGraph_All: # loop through 53 patients

    # create an empty patient dataframe to house 168 hours as signle dataframe 
    newGraph_PatientList = [] 

    for graph in patient: # loop through each hourly dataframe
        newGraph_PatientList.append(graph)# combine each hour dataframe into single dataframe
    # append newGraph_PatientFinal (single dataframe) to list of each patient
    PatientFinal = pd.concat(newGraph_PatientList, ignore_index=True)
    newGraph_AllFinal.append(PatientFinal)

for patient in newGraph_AllFinal:
    print(patient)

    
# %% 
# better version for now : 

#  -------------------- CALCULATING AUC, CORRECT VERSION FOR DAYS  ---------------------


# tuple_day_threshold = [(0, 24), (24, 48), (48, 72), (72, 96), (96, 120), (120, 144), (144, 168)]
tuple_hourly = [(i, i+1) for i in range(0, 168)]
# tuple_hourly = [() for _ len()]
patient_list2 = []
hour2 = []
total_AUC = []

less_20_list = [] 

# Function to shift ICP values
def shift_icp_values(df, shift_amount):
    df['icp'] = df['icp'] - shift_amount
    df = df[df['icp'] >= 0]
    return df

def less20_area(df):
    df = df.loc[df['icp'] <= 20, ['icp', 'Time']]
    return df

# Function to calculate AUC for given thresholds
def calc_auc(df):
    results = []
    for threshold in icp_threshold: 
        data_above_threshold = df.loc[df['icp'] >= threshold, ['icp', 'Time']]
        data_above_threshold = shift_icp_values(data_above_threshold, threshold)
        if not data_above_threshold.empty:
            x = data_above_threshold['Time'].values
            y = data_above_threshold['icp'].values
            area = np.trapz(y, x)
            results.append(area)
        else:
            results.append(0)
    return results

# patient_list = []
auc_list = [[] for _ in range(len(icp_threshold))]
test_list = []


# combine newGraph dataframe for each patient (newGraph_All will sitll have 
# 53 patients, but each patient represents 168 hours as one dataframe, 
# not 168 dataframes representing hours

for patient_df in newGraph_AllFinal:
    # Ensure df is not a Series and not empty
    if not isinstance(patient_df, pd.Series) and (len(patient_df) != 0):
        count = 0
        patient_id = patient_df['patientunitstayid'].iloc[0]
        
        patient_df = patient_df.loc[patient_df['Time'] >= 0]
        patient_df = patient_df.sort_values(by=['Time'])
        for min_time, max_time in tuple_hourly:
            day_df = patient_df.loc[(patient_df['Time'] >= min_time) & (patient_df['Time'] < max_time)]
            
            # now that day df is working with the current day; 
            if(day_df.empty):
                patient_list2.append(patient_id)
                # all values in auc_list[i] append 0 
                for i in range(len(auc_list)):
                    auc_list[i].append(0)
                count += 1
                hour2.append(count)
                test_list.append(0)
                less_20_list.append(0)
                continue

            # Append to the list         
            patient_list2.append(patient_id)
            auc_result = calc_auc(day_df)

            for i in range(len(icp_threshold)):
                auc_list[i].append(auc_result[i])
            test_list.append(np.trapz(day_df['icp'], day_df['Time']))

            test_area = less20_area(day_df)
            less_20_list.append(np.trapz(test_area['icp'], test_area['Time']))

            count += 1
            hour2.append(count)
        

# Create the DataFrame

data = {
    'patientunitstayid': patient_list2, 
    'Hour' : hour2,
    '>20 auc': auc_list[0],
    '>25 auc': auc_list[1],
    '>30 auc': auc_list[2],
    '>35 auc': auc_list[3], 
    'Total (tested)': test_list,
    '<20' : less_20_list
}

megaAUC_DF = pd.DataFrame(data)
megaAUC_DF.head(1000000000000000000000000000000000000000000000000)


#%% finding the mean/median for hr 1-24
df_24h = megaAUC_DF[megaAUC_DF['Hour'] <= 24]

# Group by patient_id and calculate the mean total_auc for the first 24 hours
df_24h_sum_auc = df_24h.groupby('patientunitstayid')['Total (tested)'].sum().reset_index()
df_24h_sum_auc.rename(columns={'Total (tested)': 'sum_auc_24h'}, index=str, inplace=True)


df_24h_nonzero_auc = df_24h[df_24h['Total (tested)'] != 0]
df_24h_nonzero_hours = df_24h_nonzero_auc.groupby('patientunitstayid')['Hour'].count().reset_index()
df_24h_nonzero_hours.rename(columns={'Hour': 'nonzero_hours_24h'}, inplace=True)


df_24h_nonzero_median = df_24h_nonzero_auc.groupby('patientunitstayid')['Total (tested)'].median().reset_index()
df_24h_nonzero_median.rename(columns={'Total (tested)': 'Median first 24 hours'}, inplace=True)

df_merged_test = pd.merge(df_24h_sum_auc, df_24h_nonzero_hours, on='patientunitstayid')
df_merged_test['Baseline ICP'] = df_merged_test['sum_auc_24h'] / df_merged_test['nonzero_hours_24h']
df_merged = pd.merge(df_merged_test, df_24h_nonzero_median, on='patientunitstayid')

df_merged = df_merged.drop(columns=['sum_auc_24h'], axis=1)
df_merged = df_merged.drop(columns=['nonzero_hours_24h'], axis=1)

df_merged.head(100000000000)
# %%

ids_to_add = [3222024, 3212405]
new_rows = pd.DataFrame({
    'patientunitstayid': ids_to_add,
    'Baseline ICP': np.nan,
    'Median first 24 hours': np.nan
})

df_merged = pd.concat([df_merged, new_rows], ignore_index=True)
df_merged = df_merged.sort_values(by='patientunitstayid').reset_index(drop=True)

df_merged.head(1000000)


# %%

print(df_merged['patientunitstayid'].unique())
print(megaStats_DF['patientunitstayid'].unique())

print(set(megaAUC_DF['patientunitstayid'].unique()) - set(megaStats_DF['patientunitstayid'].unique()))
# %%
'''
   .-.
  /'v'\    - START MERGING DATASETS UNDER HERE
 (/   \)             ONLY WORRYING THE SECOND OF THOSE TWO 
='="="===< 
mrf|_|

'''


# %%

# Now create the final dataframe woooo 
data = {
    'patientunitstayid': patient_list2, 
    'Hour' : hour2,
    'Total AUC for Hour': test_list,
    '>20 auc': auc_list[0],
    '>25 auc': auc_list[1],
    '>30 auc': auc_list[2],
    '>35 auc': auc_list[3], 
    # '<20' : less_20_list,
    'num_spike_20' : num_spike_20,
    'num_spike_25' : num_spike_25,
    'num_spike_30' : num_spike_30,
    'num_spike_35' : num_spike_35
}

AUC_RESULTS = pd.DataFrame(data)
AUC_RESULTS = pd.merge(AUC_RESULTS, df_merged, on='patientunitstayid')
cols_to_move = ['Baseline ICP', 'Median first 24 hours']
new_order = cols_to_move + [col for col in AUC_RESULTS.columns if col not in cols_to_move]
AUC_RESULTS = AUC_RESULTS[new_order]

AUC_RESULTS = AUC_RESULTS.drop(columns=['Median first 24 hours'])

cols_to_move = ['patientunitstayid', 'Hour']
new_order = cols_to_move + [col for col in AUC_RESULTS.columns if col not in cols_to_move]
AUC_RESULTS = AUC_RESULTS[new_order]

AUC_RESULTS.head(1000000000000000000000000000000000000)

# %%
AUC_RESULTS = AUC_RESULTS[(AUC_RESULTS['Hour'] >= 24) & (AUC_RESULTS['Hour'] <= 72)]

AUC_RESULTS.head(10000000)
# %%
AUC_RESULTS.to_csv('AUC_RESULTS_hourly.csv')

# %%

