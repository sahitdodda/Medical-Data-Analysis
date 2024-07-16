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
    # print(mno.matrix(dt, figsize = (20,6)))
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
    

# %% Plotting ICP thresholds to prepare for AUC

# other lists
patient_list = []
icp_list = []
time_list = []


# creates a list of a list of each patient's icp points
plotPoints_List = []
for patient_id in vitalsP_DF_patientIDs: 
    # holds original set of points
    if vitalsP_imputed_DF_copy.loc[vitalsP_imputed_DF_copy['patientunitstayid'] == patient_id, ['patientunitstayid', 'Time', 'icp']].empty:
        continue
    plotPoints = vitalsP_imputed_DF_copy.loc[vitalsP_imputed_DF_copy['patientunitstayid'] == patient_id, ['patientunitstayid', 'Time', 'icp']]
    plotPoints_List.append(plotPoints)


# creates a list of a list of each patient's NEW icp points
plotPointsNew_List = []
spikeStats_List = []
spikeCount_List = []

now = None
next = None
thresholds = [20, 25, 30, 35] # threshold list
t20 = False
t25 = False
t30 = False
t35 = False
t_cond = [t20, t25, t30, t35] # conditions

spike_Count = 0
spike_SE_List = []
spike_Duration_List = []

# trying list methodology 
count_patient = 0
# iterate through graphs
for pointsList in plotPoints_List: 
    count_patient += 1
    print(f"Patient ID: {pointsList['patientunitstayid'].iloc[0]} and # {count_patient}")
    plotPointsNew = []
    for i in range(len(pointsList)-1):
        # goes through each graph's points indiv., appends the list in order
        now = {'Time': pointsList['Time'].iloc[i], 'icp': pointsList['icp'].iloc[i]}

        patient = pointsList['patientunitstayid'].iloc[i]

        patient_list.append(pointsList['patientunitstayid'].iloc[i])
        icp_list.append(pointsList['icp'].iloc[i])
        time_list.append(pointsList['Time'].iloc[i])

        # plotPointsNew = pd.concat([pointsList, new_row], ignore_index=True)
        # plotPointsNew.append(now) # add the next point to the list

        next = {'Time': pointsList['Time'].iloc[i+1], 'icp': pointsList['icp'].iloc[i+1]}

        # if both points are the same, no need to add a new point
        if(now['icp'] == next['icp']):
            continue
        # takes care if a point goes over multiple thresholds
        for i in range(len(thresholds)):
            if((now['icp'] < thresholds[i] and thresholds[i] < next['icp']) or (now['icp'] > thresholds[i] and thresholds[i] > next['Time'])): # only counts points if NOT exactly threshold
                t_cond[i] = True
                # print('set condition')
        
        # positive or negative slope 
        slope = (next['icp'] - now['icp']) / (next['Time'] - now['Time']) # pos. or neg. slope
        # print(slope)
        # crosses 20
        if(t_cond[0]):
            x = ((20-now['icp'])/slope) + now['Time'] # time where it crosses threshold
            
            patient_list.append(pointsList['patientunitstayid'].iloc[i])
            icp_list.append(20)
            time_list.append(x)
            if(slope>0):
                spike_Start = x
                # spike_Count += 1
            else:
                spike_End = x
                spike_SE_List.append((spike_Start, spike_End))
                spike_Duration = spike_End - spike_Start
                spike_Duration_List.append(spike_Duration)
            # print(f"20 with the now icp: {now['icp']} and the next icp: {next['icp']}")
            # plotPointsNew[i].append({'Time': x, 'icp': 20})
        # crosses 25
        if(t_cond[1]):
            x = ((25-now['icp'])/slope) + now['Time'] # time where it crosses threshold
            patient_list.append(pointsList['patientunitstayid'].iloc[i])
            icp_list.append(25)
            time_list.append(x)
        # crosses 30
        if(t_cond[2]):
            x = ((30-now['icp'])/slope) + now['Time'] # time where it crosses threshold
            patient_list.append(pointsList['patientunitstayid'].iloc[i])
            icp_list.append(30)
            time_list.append(x)
            # plotPointsNew[i].append({'Time': x, 'icp': 30})
        # crosses 35
        if(t_cond[3]):
            x = ((35-now['icp'])/slope) + now['Time'] # time where it crosses threshold
            patient_list.append(pointsList['patientunitstayid'].iloc[i])
            icp_list.append(35)
            time_list.append(x)
            # plotPointsNew[i].append({'Time': x, 'icp': 35})

        # reset condiitons
        t_cond[0] = False
        t_cond[1] = False
        t_cond[2] = False
        t_cond[3] = False
        
    data = {
        'patientunitstayid' : patient_list, 
        'icp' : icp_list,
        'Time' : time_list,
    }
    #dbdfbffdbdfbdfbdfbdfbfdbfdbdfbdfbfdbdfbdfbdfbdfbdfbdfbdfbdfbdfbdfbdfbdfbdfbdfbdfbdfbdfbdfbfdbfdbdfbfdbdfb
    # 'Spike Count' : spike_Count
    # 'Spike Start/End' : spike_SE_List
    # 'Spike Duration' : spike_Duration_List

    x1_time = [t[0] for t in spike_SE_List]
    x2_time = [t[1] for t in spike_SE_List]
    patient2_list = [patient for t in spike_SE_List]
    spikeCount_List.append(len(patient2_list))

    data2 = {
        'patientunitstayid' : patient2_list,
        'start_time' : x1_time,
        'end_time' : x2_time,
        'spike_duration' : spike_Duration_List
    }

    stat_DF = pd.DataFrame(data2)
    spikeStats_List.append(stat_DF)

    patient_list = []
    icp_list = []
    time_list = []
    spike_Count = 0
    plotPointsNew = pd.DataFrame(data)

    plotPointsNew_List.append(plotPointsNew)

# print
# for plotPointsNew in plotPointsNew_List:
#     print(plotPointsNew.head())

# for df in spikeStats_List:
#     print(df.head())

# %% 
# now append all spike df's in the list to a new dataframe 

spike_StatsDF = pd.concat(spikeStats_List, ignore_index=True)
print(spike_StatsDF)
print('\n\n\n\n')

print(f'Spike count list {spikeCount_List}')
print(f'Spike count list length {len(spikeCount_List)}')
# %%

# Thresholds for ICP ranges
thresholds = [20, 25, 30, 35]
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
    for threshold in thresholds: 
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

patient_list = []
auc_list = [[] for _ in range(len(thresholds))]
test_list = []

for df in plotPointsNew_List:
    # Ensure df is not a Series and not empty
    if not isinstance(df, pd.Series) and (len(df) != 0):
        patient_id = df['patientunitstayid'].iloc[0]
        
        # Append to the list         
        patient_list.append(patient_id)

        auc_result = calc_auc(df)

        for i in range(len(thresholds)):
            auc_list[i].append(auc_result[i])
        test_list.append(np.trapz(df['icp'], df['Time']))

        test_area = less20_area(df)
        less_20_list.append(np.trapz(test_area['icp'], test_area['Time']))
        
# Create the DataFrame
data = {
    'patientunitstayid': patient_list, 
    '>20': auc_list[0],
    '>25': auc_list[1],
    '>30': auc_list[2],
    '>35': auc_list[3], 
    'Total (tested)': test_list,
    '<20' : less_20_list
}

df_auc_ranges = pd.DataFrame(data)

print(f"Number of plot points: {len(plotPointsNew_List)}")
df_auc_ranges.head(10000000000000000000000000000000000000000)


# %%
'''
   .-.
  /'v'\    - START MERGING DATASETS UNDER HERE
 (/   \)
='="="===< 
mrf|_|
'''

# %%
# append spikeCountList as a column afterwards

apache_DF = pd.read_csv('apache_results.csv')
diag_DF = pd.read_csv('diagP_results.csv')

df_patient_STATS = pd.merge( df_auc_ranges, df_range, on='patientunitstayid')
df_patient_STATS = pd.merge(df_patient_STATS, apache_DF, on='patientunitstayid')

# df_patient_STATS['spike_Count'] = spikeCount_List

df_patient_STATS.head(100000000000000)
