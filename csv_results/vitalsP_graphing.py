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
    

# %% Plotting ICP thresholds to prepare for AUC

# other lists
# patient_list = []
# icp_list = []
# time_list = []


# # creates a list of a list of each patient's icp points
# plotPoints_List = []
# for patient_id in vitalsP_DF_patientIDs: 
#     # holds original set of points
#     if vitalsP_imputed_DF_copy.loc[vitalsP_imputed_DF_copy['patientunitstayid'] == patient_id, ['patientunitstayid', 'Time', 'icp']].empty:
#         continue
#     plotPoints = vitalsP_imputed_DF_copy.loc[vitalsP_imputed_DF_copy['patientunitstayid'] == patient_id, ['patientunitstayid', 'Time', 'icp']]
#     plotPoints_List.append(plotPoints)


# # creates a list of a list of each patient's NEW icp points
# plotPointsNew_List = []
# spikeStats_List = []
# spikeCount_List = []

# now = None
# next = None
# thresholds = [20, 25, 30, 35] # threshold list
# t20 = False
# t25 = False
# t30 = False
# t35 = False
# t_cond = [t20, t25, t30, t35] # conditions

# spike_Count = 0
# spike_SE_List = []
# spike_Duration_List = []

# # trying list methodology 
# count_patient = 0
# # iterate through graphs
# for pointsList in plotPoints_List: 
#     count_patient += 1
#     print(f"Patient ID: {pointsList['patientunitstayid'].iloc[0]} and # {count_patient}")
#     plotPointsNew = []
#     for i in range(len(pointsList)-1):
#         # goes through each graph's points indiv., appends the list in order
#         now = {'Time': pointsList['Time'].iloc[i], 'icp': pointsList['icp'].iloc[i]}

#         patient = pointsList['patientunitstayid'].iloc[i]

#         patient_list.append(pointsList['patientunitstayid'].iloc[i])
#         icp_list.append(pointsList['icp'].iloc[i])
#         time_list.append(pointsList['Time'].iloc[i])

#         # plotPointsNew = pd.concat([pointsList, new_row], ignore_index=True)
#         # plotPointsNew.append(now) # add the next point to the list

#         next = {'Time': pointsList['Time'].iloc[i+1], 'icp': pointsList['icp'].iloc[i+1]}

#         # if both points are the same, no need to add a new point
#         if(now['icp'] == next['icp']):
#             continue
#         # takes care if a point goes over multiple thresholds
#         for i in range(len(thresholds)):
#             if((now['icp'] < thresholds[i] and thresholds[i] < next['icp']) or (now['icp'] > thresholds[i] and thresholds[i] > next['Time'])): # only counts points if NOT exactly threshold
#                 t_cond[i] = True
#                 # print('set condition')
        
#         # positive or negative slope 
#         slope = (next['icp'] - now['icp']) / (next['Time'] - now['Time']) # pos. or neg. slope
#         # print(slope)
#         # crosses 20
#         if(t_cond[0]):
#             x = ((20-now['icp'])/slope) + now['Time'] # time where it crosses threshold
            
#             patient_list.append(pointsList['patientunitstayid'].iloc[i])
#             icp_list.append(20)
#             time_list.append(x)
#             if(slope>0):
#                 spike_Start = x
#                 # spike_Count += 1
#             else:
#                 spike_End = x
#                 spike_SE_List.append((spike_Start, spike_End))
#                 spike_Duration = spike_End - spike_Start
#                 spike_Duration_List.append(spike_Duration)
#             # print(f"20 with the now icp: {now['icp']} and the next icp: {next['icp']}")
#             # plotPointsNew[i].append({'Time': x, 'icp': 20})
#         # crosses 25
#         if(t_cond[1]):
#             x = ((25-now['icp'])/slope) + now['Time'] # time where it crosses threshold
#             patient_list.append(pointsList['patientunitstayid'].iloc[i])
#             icp_list.append(25)
#             time_list.append(x)
#         # crosses 30
#         if(t_cond[2]):
#             x = ((30-now['icp'])/slope) + now['Time'] # time where it crosses threshold
#             patient_list.append(pointsList['patientunitstayid'].iloc[i])
#             icp_list.append(30)
#             time_list.append(x)
#             # plotPointsNew[i].append({'Time': x, 'icp': 30})
#         # crosses 35
#         if(t_cond[3]):
#             x = ((35-now['icp'])/slope) + now['Time'] # time where it crosses threshold
#             patient_list.append(pointsList['patientunitstayid'].iloc[i])
#             icp_list.append(35)
#             time_list.append(x)
#             # plotPointsNew[i].append({'Time': x, 'icp': 35})

#         # reset condiitons
#         t_cond[0] = False
#         t_cond[1] = False
#         t_cond[2] = False
#         t_cond[3] = False
        
#     data = {
#         'patientunitstayid' : patient_list, 
#         'icp' : icp_list,
#         'Time' : time_list,
#     }
#     #dbdfbffdbdfbdfbdfbdfbfdbfdbdfbdfbfdbdfbdfbdfbdfbdfbdfbdfbdfbdfbdfbdfbdfbdfbdfbdfbdfbdfbdfbfdbfdbdfbfdbdfb
#     # 'Spike Count' : spike_Count
#     # 'Spike Start/End' : spike_SE_List
#     # 'Spike Duration' : spike_Duration_List

#     x1_time = [t[0] for t in spike_SE_List]
#     x2_time = [t[1] for t in spike_SE_List]
#     patient2_list = [patient for t in spike_SE_List]
#     spikeCount_List.append(len(patient2_list))

#     data2 = {
#         'patientunitstayid' : patient2_list,
#         'start_time' : x1_time,
#         'end_time' : x2_time,
#         'spike_duration' : spike_Duration_List
#     }

#     stat_DF = pd.DataFrame(data2)
#     spikeStats_List.append(stat_DF)

#     patient_list = []
#     icp_list = []
#     time_list = []
#     spike_Count = 0
#     plotPointsNew = pd.DataFrame(data)

#     plotPointsNew_List.append(plotPointsNew)

#%%

# ʕ •ᴥ•ʔ 
# ʕ っ•ᴥ•ʔっ
# ʕ ᵔᴥᵔ  ʔ

# make a now dictionary 
    # then grab the corresponding patient 
# default append at the beginning. 
# set up the 'next' dictionary. 

# check if the major wil be equal to the next and force continue if the case. 
# now check if the point goes over many thresholds using the corresponding boolean lists. 
# calculate the slope, and then check all conditions within the list. append as necessary within each cond. 

# then reset all conditions using a list comprehension.
# make a dataframe of the results and append to the list. 
# repeat 


# icp_threshold = [20, 25, 30, 35, None]
# t_cond_icp = [False for _ in range(len(icp_threshold))]
# Day_threshold = [24, 48, 72, 96, 120, 144, 168, None]
# t_cond_Day = [False for _ in range(len(Day_threshold))]

# all_thresholds = [
#     (('icp', 'Time'), icp_threshold), 
#     (('Time', 'icp'), Day_threshold)
# ]

# def calc_thresh_val(y1, y2, x1, x2, threshold): # chose refers to icp or Time 
#     ydiff = y2-y1
#     xdiff = x2-x1
#     if xdiff == 0:
#         return y1, x1    
#     slope = ydiff/xdiff

#     new_value = (threshold-y1)/slope + x1
#     return new_value, threshold


# # assumes you are iterating through the larger list that is created with a df 
# # of each patient. 
# def find_thresholds(patient_df, thresholds, columnMajor, columnMinor, patient_list, icp_list, time_list):
#     for i in range(len(patient_df) - 1):
        
#         # for example, valueMajor may be icp. generic version of now and next 
#         valueMajor = patient_df['patientunitstayid'].iloc[i]
#         nextValueMajor = patient_df[valueMajor].iloc[i]

#         valueMinor = patient_df[valueMinor].iloc[i + 1]
#         nextValueMinor = patient_df[valueMinor].iloc[i + 1]

#         patient = patient_df['patientunitstayid'].iloc[i]
#         patient_list.append(patient)
#         icp_list.append(patient_df['icp'].iloc[i])
#         time_list.append(patient_df['Time'].iloc[i])

#         if(valueMajor == nextValueMajor):
#             continue
        

#         for i in range(len(thresholds)):
#             if(valueMajor < thresholds[i] and thresholds[i] > nextValueMajor) or (valueMajor > thresholds[i] and thresholds[i] < nextValueMajor):
#                 if(columnMajor == 'icp'):
#                     t_cond_icp[i] = True
#                 else: 
#                     t_cond_Day[i]

        
#         for i, threshold in enumerate(thresholds):

#             # same thing for either branch
#             if columnMajor == 'icp':
#                 if t_cond_icp[i]:
#                     major_value, minor_value = calc_thresh_val(valueMajor, nextValueMajor, valueMinor, nextValueMinor, threshold)
#                     patient_list.append(patient_df['patientunitstayid'].iloc[0])
#                     if columnMajor == 'icp':
#                         icp_list.append(minor_value)
#                         time_list.append(major_value)
#                     else:
#                         icp_list.append(major_value)
#                         time_list.append(minor_value)
#             else:
#                 if t_cond_Day[i]:
#                     major_value, minor_value = calc_thresh_val(valueMajor, nextValueMajor, valueMinor, nextValueMinor, threshold)
#                     patient_list.append(patient_df['patientunitstayid'].iloc[0])
#                     if columnMajor == 'icp':
#                         icp_list.append(minor_value)
#                         time_list.append(major_value)
#                     else:
#                         icp_list.append(major_value)
#                         time_list.append(minor_value)

#         # now reset conditions

#         t_cond_icp = [False for _ in range(len(t_cond_icp))]
#         t_cond_Day = [False for _ in range(len(t_cond_Day))]

#     # data = {
#     #     'patientunitstayid' : patient_list,
#     #     'icp' : icp_list, 
#     #     'Time' : time_list 
#     # }

#     # new_patient_df = pd.DataFrame(data)

#     return patient_list, icp_list, time_list 


# data_sample = {
#     'patientunitstayid' : [1, 1, 1, 1, 1],
#     'icp' : [17, 19, 31, 1, 3],
#     'Time' : [2, 4, 16, 32, 18]
# }
# data_sample = pd.DataFrame(data_sample)

# a = []
# b = []
# c = []
# patient_list, icp_list, time_list = find_thresholds(data_sample, icp_threshold,  'icp', 'Time', a, b, c)




#%%
icp_threshold = [20, 25, 30, 35]
t_cond_icp = [False for _ in range(len(icp_threshold))]
Day_threshold = [24, 48, 72, 96, 120, 144, 168]
t_cond_Day = [False for _ in range(len(Day_threshold))]

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

# for major ICP
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
        patient_list, time_list, icp_list = find_thresholds(patient_id, Day_threshold, 'Time', 'icp', patient_list, icp_list, time_list)
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



# %% Plotting ICP thresholds to prepare for AUC
list = [0,0,0]
for i in range(len(list)):
    print(i)


# %%




# %%

# -------------------------------------- Pseudocode --------------------------------------

# all patients (node in vitalsP_LL)
    # each patient (node.data (vitalsP for that ID))
        # each day (create 7 days from node.data)
            # each point (find new points, spike count, spike times, spike durations)

                # save new points, spike count, spike times, spike durations into a dataframe
                # save dataframe into a totalDay_List (day 1-7)
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


spikeStats_All = []
newGraph_All = []


node = vitalsP_imputed_LL.head
numNode = 0
while node: # by patient
    print(f'Node {numNode}')
    df = node.data # datafarame per ID 
    ID = df['patientunitstayid'].iloc[0] # patient ID
    # saves OG points (day 1-7)
    day1_Graph = df.loc[df['Time'] < 24]
    day2_Graph = df.loc[(df['Time'] >= 24) & (df['Time'] < 48)]
    day3_Graph = df.loc[(df['Time'] >= 48) & (df['Time'] < 72)]
    day4_Graph = df.loc[(df['Time'] >= 72) & (df['Time'] < 96)]
    day5_Graph = df.loc[(df['Time'] >= 96) & (df['Time'] < 120)]
    day6_Graph = df.loc[(df['Time'] >= 120) & (df['Time'] < 144)]
    day7_Graph = df.loc[(df['Time'] >= 144) & (df['Time'] < 168)]
    # saves into a list
    DPList = [day1_Graph, day2_Graph, day3_Graph, day4_Graph, day5_Graph, day6_Graph, day7_Graph]
    # reset day for next patient
    day = 0
    newGraph_Patient = []
    spikeStats_Patient = []

    lastDay = 0
    lastHour = 0
    first = False

    for graph in DPList: # each graph, by day
        day += 1
        newGraph = pd.DataFrame(columns=['patientunitstayid', 'Day', 'Time', 'icp']) # holds points
        spikeStats = pd.DataFrame(columns=['patientunitstayid', 'Day', 'lastDay', 'spikes', 'spikeStarts', 'spikeEnds']) # holds lists
        
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
            newRow = pd.DataFrame({'patientunitstayid': ID, 'Day': day, 'Time': nT, 'icp': nI}, index=[0])
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
                newRow = pd.DataFrame({'patientunitstayid': ID, 'Day': day, 'Time': x, 'icp': 20}, index=[0])
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
                newRow = pd.DataFrame({'patientunitstayid': ID, 'Day': day, 'Time': x, 'icp': 25}, index=[0])
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
                newRow = pd.DataFrame({'patientunitstayid': ID, 'Day': day, 'Time': x, 'icp': 30}, index=[0])
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
                newRow = pd.DataFrame({'patientunitstayid': ID, 'Day': day, 'Time': x, 'icp': 35}, index=[0])
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
                newRow = pd.DataFrame({'patientunitstayid': ID, 'Day': day, 'Time': nxT, 'icp': nxI}, index=[0])
                # Check if newRow is not empty or all-NA before concatenation
                if not newRow.isna().all(axis=None):
                    newGraph = pd.concat([newGraph, newRow], ignore_index=True)


                # manually count when the graphs crosses a 
                # threshold AND include values 
                # where the current point is on the 
                # threshold and passes into the threshol

            # reset condiitons to prep for next point
            t_cond[0] = False
            t_cond[1] = False
            t_cond[2] = False
            t_cond[3] = False
    
        # append newGraph to total Patient list
        newGraph_Patient.append(newGraph)
        if(newGraph.empty and first == False):
            lastDay = day
            lastHour = (df['Time'].iloc[-1]) % 24
            first = True

        # WE ONL WANT STATS SPIKES 

        # append statsList to total Patient list
        spikes = [spike_Count20, spike_Count25, spike_Count30, spike_Count35]
        start = [start20, start25, start30, start35]
        end = [end20, end25, end30, end35]

        spikeStats = pd.DataFrame({'patientunitstayid': ID, 'Day': day, 'spikes': spikes, 'lastDay': lastDay, 'lastHour': lastHour, 'spikeStarts': start, 'spikeEnds': end})
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

        plt.figure(figsize=(10, 6))
        # Plotting the graph
        plt.plot(newGraph['Time'], newGraph['icp'], marker='o', linestyle='-')
        # Adding horizontal lines at specific icp values
        plt.axhline(y=20, color='r', linestyle='--', label='Threshold 20')
        plt.axhline(y=25, color='g', linestyle='--', label='Threshold 25')
        plt.axhline(y=30, color='b', linestyle='--', label='Threshold 30')
        plt.axhline(y=35, color='y', linestyle='--', label='Threshold 35')
        # Adding title and labels
        plt.title(f'{ID}, Day {day}: ICP vs Time')
        plt.xlabel('Time')
        plt.ylabel('ICP')
        # Adding legend
        plt.legend()
        # Display the plot
        plt.show()

        print(f'Number of days: {len(spikeStats_Patient)}')
        print(f'Number of days: {len(newGraph_Patient)}')

        # reset conditions for next day
        spike_Count20, spike_Count25, spike_Count30, spike_Count35 = 0, 0, 0, 0
        start20, start25, start30, start35 = [], [], [], []
        end20, end25, end30, end35 = [], [], [], []


    spikeStats_All.append(spikeStats_Patient)
    newGraph_All.append(newGraph_Patient)
    lastDay = 0
    first = False

    numNode += 1
    node = node.next


# ------------------------------------------------- PRINTING ALL VALUES -------------------------------------------------

print(f'Number of patients: {len(spikeStats_All)}')
print(f'Number of patients: {len(newGraph_All)}')

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



icp_time = 8


for list in spikeStats_All: # cycles through each patient
    
    for i in range(len(list)): # cycles through each day

        # patientID
        patientID = list[i]['patientunitstayid'].iloc[0]
        # time period
        day = i+1
        # total time icp was measured
        if(day < list[i]['lastDay'].values[0]):
            icp_time = 24 # rdoesn't run
        if(day == list[i]['lastDay'].values[0]):
            icp_time = list[i]['lastHour'].values[0] # good
        if(day > list[i]['lastDay'].values[0]):
            icp_time = 0
        # spike stats
        spike_Count20 = list[i]['spikes'][0]
        spike_Count25 = list[i]['spikes'][1]
        spike_Count30 = list[i]['spikes'][2]
        spike_Count35 = list[i]['spikes'][3]
        # append to the list
        patient_list.append(patientID)
        time_period.append(day)
        icp_period.append(icp_time)
        num_spike_20.append(spike_Count20)
        num_spike_25.append(spike_Count25)
        num_spike_30.append(spike_Count30)
        num_spike_35.append(spike_Count35)

        

data = {
    'patientunitstayid' : patient_list, 
    'Day' : time_period, 
    'icp_period' : icp_period,
    'num_spike_20' : num_spike_20, 
    'num_spike_25' : num_spike_25,
    'num_spike_30' : num_spike_30,
    'num_spike_35' : num_spike_35
}

megaStats_DF = pd.DataFrame(data)

megaStats_DF.head(1000000000000000000000000000000000000)


 # %%

#  -------------------- CALCULATING AUC, CORRECT VERSION FOR DAYS  ---------------------

tuple_day_threshold = [(0, 24), (24, 48), (48, 72), (72, 96), (96, 120), (120, 144), (144, 168)]

patient_list2 = []
day2 = []
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

for patient_df in plotPointsNew_List:
    # Ensure df is not a Series and not empty
    if not isinstance(patient_df, pd.Series) and (len(patient_df) != 0):
        count = 0
        patient_id = patient_df['patientunitstayid'].iloc[0]
        
        patient_df = patient_df.loc[patient_df['Time'] >= 0]
        patient_df = patient_df.sort_values(by=['Time'])
        for min_time, max_time in tuple_day_threshold:
            day_df = patient_df.loc[(patient_df['Time'] >= min_time) & (patient_df['Time'] < max_time)]
            
            # now that day df is working with the current day; 
            if(day_df.empty):
                patient_list2.append(patient_id)
                # all values in auc_list[i] append 0 
                for i in range(len(auc_list)):
                    auc_list[i].append(0)
                count += 1
                day2.append(count)
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
            day2.append(count)
        

# Create the DataFrame

data = {
    'patientunitstayid': patient_list2, 
    'Day' : day2,
    '>20 auc': auc_list[0],
    '>25 auc': auc_list[1],
    '>30 auc': auc_list[2],
    '>35 auc': auc_list[3], 
    'Total (tested)': test_list,
    '<20' : less_20_list
}

megaAUC_DF = pd.DataFrame(data)
megaAUC_DF.head(1000000000000000000000000000000000000)

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
    'Day' : day2,
    'icp_period' : icp_period,
    'Total AUC for Day': test_list,
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
AUC_RESULTS.head(1000000000000000000000000000000000000)

AUC_RESULTS.to_csv('AUC_RESULTS.csv')