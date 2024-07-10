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
# %% Imputation


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

print(node.to_string()) # print orig dataframe
print("systemicmean left", dt['systemicmean'].isna().sum()) # num. of nan's left in systemicmean

#              -------------- Imputation --------------


# iterate through each the missing times with variable 'spot' -> look fwd/bwd -> impute based on condiitons
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

# %% (imputation)

# systemicmeanImpute(vitalsP_DF)
# systemicsystoic
# heartrate
# respiration
# spo2
# temperature

# %% (Visualization) streamlit graphs

#                           ---------- All ICP v Time ----------
st.titlfsdfsdfe('All ICP values vs Time')

fig = go.Figure()
for patient_id in vitalsP_DF['patientunitstayid'].unique():
    patient_data = vitalsP_DF[vitalsP_DF['patientunitstayid'] == patient_id]
    fig.add_trace(go.Scatter(x=patient_data['Time'], y=patient_data['icp'], mode='lines', name=f'Patient {patient_id}'))
fig.update_layout(title='All ICP values vs Time', xaxis_title='Time', yaxis_title='ICP Value', hovermode='closest')
fig.update_yaxes(range=[0, 50])

fig.show()

#               ---------- ICP of Alive/Expired patients, day 1-3 histograms ----------

# setting variables for following methods
count20_25, count25_30, count30_, count25_30, count30_ = 0, 0, 0, 0, 0
within_24h, within_48h, within_72h = aliveID_DF[aliveID_DF['Time'] <= 24], aliveID_DF[aliveID_DF['Time'] <= 48], aliveID_DF[aliveID_DF['Time'] <= 72]
listAliveWithin = [within_24h, within_48h, within_72h]
expired_within_24h, expired_within_48h, expired_within_72h = expiredID_DF[expiredID_DF['Time'] <= 24], expiredID_DF[expiredID_DF['Time'] <= 48], expiredID_DF[expiredID_DF['Time'] <= 72]
listExpWithin = [expired_within_24h, expired_within_48h, expired_within_72h]
listChoose = [listAliveWithin, listExpWithin, aliveID_DF, expiredID_DF]
# setting the counts
def setCount(graph, day):
    if(day != 3):
        count20_25 = listChoose[graph][day][(listChoose[graph][day]['icp'] >= 20) & (listChoose[graph][day]['icp'] <= 25)].shape[0]
        count25_30 = listChoose[graph][day][(listChoose[graph][day]['icp'] >= 25) & (listChoose[graph][day]['icp'] <= 30)].shape[0]
        Count30_ = listChoose[graph][day][(listChoose[graph][day]['icp'] >= 30)].shape[0]
        count = listChoose[graph][day]['patientunitstayid'].unique().shape[0]
        return count, count20_25, count25_30, Count30_
    else:
        count20_25 = listChoose[graph][(listChoose[graph]['icp'] >= 20) & (listChoose[graph]['icp'] <= 25)].shape[0]
        count25_30 = listChoose[graph][(listChoose[graph]['icp'] >= 25) & (listChoose[graph]['icp'] <= 30)].shape[0]
        Count30_ = listChoose[graph][(listChoose[graph]['icp'] >= 30)].shape[0]
        count = listChoose[graph]['patientunitstayid'].unique().shape[0]
        return count, count20_25, count25_30, Count30_
# method to print measurements
def print_icp_counts(countAlive, count20_25, count25_30, count30_):
    print("# of Alive Patients: ", countAlive)
    print("# spikes w ICP 20-25: ", count20_25)
    print("# spikes w ICP 25-30: ", count25_30)
    print("# spikes w ICP 30+: ", count30_)

titleList = ['Alive', 'Expired']
# graphing histograms
def histogram(graph, day):
    with st.expander(f'Day {day} ICP Histogram for {titleList[graph]} Patients'):
        # Select only the 'icp' column values from the new dataframe
        withinTimeFrame = listChoose[graph][day]['icp']
        # Create a histogram
        fig = go.Figure(data=[go.Histogram(x=withinTimeFrame, xbins=dict(start=0, end=50, size=1))])
        fig.update_layout(title=f'Day {day} ICP Histogram for {titleList[graph]} Patients',
                      xaxis_title='ICP Value',
                      yaxis_title='Count',
                      bargap=0.2)
   
    fig.show()
    
# Alive patients graph
st.title('Interactive ICP of Alive patients')
fig = go.Figure()
for patient_id in aliveID_DF['patientunitstayid'].unique():
    patient_data = aliveID_DF[aliveID_DF['patientunitstayid'] == patient_id]
    fig.add_trace(go.Scatter(x=patient_data['Time'], y=patient_data['icp'], mode='lines', name=f'Patient {patient_id}'))
fig.update_layout(title='ICP Values of Alive Patients', xaxis_title='Time', yaxis_title='ICP Value', hovermode='closest')
fig.update_yaxes(range=[5, 55])

fig.show()
# print the counts
count, count20_25, count25_30, Count30_ = setCount(2, 3)
print_icp_counts(count, count20_25, count25_30, Count30_)

# Day 1, Alive Histogram
histogram(0, 0)
count, count20_25, count25_30, Count30_ = setCount(0, 0)
print_icp_counts(count, count20_25, count25_30, Count30_)
# Day 2, Alive Histogram
histogram(0, 1)
count, count20_25, count25_30, Count30_ = setCount(0, 1)
print_icp_counts(count, count20_25, count25_30, Count30_)
# Day 3, Alive Histogram
histogram(0, 2)
count, count20_25, count25_30, Count30_ = setCount(0, 2)
print_icp_counts(count, count20_25, count25_30, Count30_)


# Expired patients graph
st.title('Interactive ICP of Expired patients')

fig = go.Figure()

for patient_id in expiredID_DF['patientunitstayid'].unique():
    patient_data = expiredID_DF[expiredID_DF['patientunitstayid'] == patient_id]
    fig.add_trace(go.Scatter(x=patient_data['Time'], y=patient_data['icp'], mode='lines', name=f'Patient {patient_id}'))
fig.update_layout(title='ICP Values of Expired Patients', xaxis_title='Time', yaxis_title='ICP Value', hovermode='closest')
fig.update_yaxes(range=[5, 55])

fig.show()

count, count20_25, count25_30, Count30_ = setCount(3, 3)
print_icp_counts(count, count20_25, count25_30, Count30_)

# Day 1, Expired Histogram
histogram(1, 0)
count, count20_25, count25_30, Count30_ = setCount(1, 0)
print_icp_counts(count, count20_25, count25_30, Count30_)
# Day 2, Expired Histogram
histogram(1, 1)
count, count20_25, count25_30, Count30_ = setCount(1, 1)
print_icp_counts(count, count20_25, count25_30, Count30_)
# Day 3, Expired Histogram
histogram(1, 2)
count, count20_25, count25_30, Count30_ = setCount(1, 2)
print_icp_counts(count, count20_25, count25_30, Count30_)
# %% Vitals of each patient with ICP
st.title('Vitals of Every Patient')

tempNode = vitalsP_LL.head
while tempNode:
    dt = tempNode.data
    patient = dt.index.get_level_values('patientunitstayid').unique()
    time = dt.index.get_level_values('Time')
    with st.expander(f'Patient ID: {patient}'):
        fig = go.Figure()
        numeric_columns = dt.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            if column != 'Time':  # Exclude 'Time' as it's our x-axis
                fig.add_trace(go.Scatter(x=time, y=dt[column], mode='lines', name=column))
       
        fig.update_layout(title = f'Patient ID {patient}', xaxis_title = 'Time', yaxis_title = 'Value', hovermode = 'closest')


        # see if we need this
        # fig.update_yaxes(range=[dt[numeric_columns].min().min(), dt[numeric_columns].max().max()])
       
        if(dt.index.get_level_values('Time').max() > 72):
            fig.update_xaxes(range=[0,72])


        fig.show()
   
    tempNode = tempNode.next

# ----------------very big line break ----------------------

'''
     (\          
    (  \  /(o)\    The code below is for estimates for area    
    (   \/  ()/ /)  under the curve for various intervals.
     (   `;.))'".)      Atm, the numbers should work but aren't the most usable
      `(/////.-'
   =====))=))===()
     ///'      
    //   PjP/ejm
   '    
'''

# %% Plotting ICP thresholds to prepare for AUC

# other lists
patient_list = []
icp_list = []
time_list = []


# creates a list of a list of each patient's icp points
plotPoints_List = []
for patient_id in vitalsP_DF_patientIDs: 
    # holds original set of points
    if vitalsP_DF.loc[vitalsP_DF['patientunitstayid'] == patient_id, ['patientunitstayid', 'Time', 'icp']].empty:
        continue
    plotPoints = vitalsP_DF.loc[vitalsP_DF['patientunitstayid'] == patient_id, ['patientunitstayid', 'Time', 'icp']]
    plotPoints_List.append(plotPoints)


# creates a list of a list of each patient's NEW icp points
plotPointsNew_List = []
now = None
next = None
thresholds = [20, 25, 30, 35] # threshold list
t20 = False
t25 = False
t30 = False
t35 = False
t_cond = [t20, t25, t30, t35] # conditions

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
    patient_list = []
    icp_list = []
    time_list = []
    plotPointsNew = pd.DataFrame(data)

    plotPointsNew_List.append(plotPointsNew)

# print
for plotPointsNew in plotPointsNew_List:
    print(plotPointsNew.head())
# %% (prep code) AUC for ICP spikes
# Here we estimate the area under. Start by excluding irrelevant values
vitalsP_DF['icpSpike20to25'] = vitalsP_DF['icp'].where((vitalsP_DF['icp'] >= 20) & (vitalsP_DF['icp'] <= 25))
vitalsP_DF['icpSpike25to30'] = vitalsP_DF['icp'].where((vitalsP_DF['icp'] >= 25) & (vitalsP_DF['icp'] <= 30))
vitalsP_DF['icpSpike30to35'] = vitalsP_DF['icp'].where((vitalsP_DF['icp'] >= 30) & (vitalsP_DF['icp'] <= 35))
vitalsP_DF['icpSpike35+'] = vitalsP_DF['icp'].where(vitalsP_DF['icp'] >= 30)

# %% method for area, plotting data points
def process_icp_range(vitalsP_DF, time_col, icp_col, min_icp, max_icp):
    # Create ICP range column
    range_col = f'icpSpike{min_icp}to{max_icp}'
    vitalsP_DF[range_col] = vitalsP_DF[icp_col].where((vitalsP_DF[icp_col] >= min_icp) & (vitalsP_DF[icp_col] <= max_icp))
   
    # Prepare data
    df_clean = vitalsP_DF.reset_index()
    df_clean = df_clean[[time_col, range_col]].dropna().reset_index(drop=True)
    time_clean = df_clean[time_col].values
    icp_clean = df_clean[range_col].values
   
    # Calculate area
    area = np.trapz(icp_clean, time_clean)
   
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_clean, icp_clean)
    plt.xlabel('Time')
    plt.ylabel(f'ICP ({min_icp}-{max_icp} range)')
    plt.title(f'ICP values in {min_icp}-{max_icp} range')
   
    # Add area and data points info to plot
    total_points = len(vitalsP_DF)
    clean_points = len(time_clean)
    percentage_used = (clean_points / total_points) * 100
    plt.text(0.05, 0.95, f'Area: {area:.2f}\nData points: {clean_points}/{total_points} ({percentage_used:.2f}%)',
             transform=plt.gca().transAxes, verticalalignment='top')
   
    plt.show()
   
    return area, clean_points, total_points

# Define ICP ranges
icp_ranges = [(0, 20), (20, 25), (25, 30), (30, 40), (40, 50)]

# Process each range
results = []
for min_icp, max_icp in icp_ranges:
    area, clean_points, total_points = process_icp_range(vitalsP_DF, 'Time', 'icp', min_icp, max_icp)
    results.append({
        'range': f'{min_icp}-{max_icp}',
        'area': area,
        'clean_points': clean_points,
        'total_points': total_points,
        'percentage_used': (clean_points / total_points) * 100
    })

# Print summary
for result in results:
    print(f"ICP Range {result['range']}:")
    print(f"  Estimated area: {result['area']:.2f}")
    print(f"  Data points: {result['clean_points']}/{result['total_points']} ({result['percentage_used']:.2f}%)")
    print()
# %% VARIABLE LIST

vitalsP_DF = vitalsP_DF # original data set, fully cleaned