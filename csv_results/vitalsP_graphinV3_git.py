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
vitalsP_DF = vitalsP_DF.drop(columns=['Unnamed: 0', 'observationoffset', 'Day', 'Hour', 'systemicdiastolic', 'systemicsystolic'])
# TIME (STRING -> INTEGER)

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

vitalsP_DF = vitalsP_DF.loc[vitalsP_DF['patientunitstayid'].isin(total_patient_list)]


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
    # df_multiobj = vitalsP_MultiIndex[vitalsP_MultiIndex.index.get_level_values('patientunitstayid') == patient_id]
    dfIter = vitalsP_MultiIndex.xs(patient_id, level='patientunitstayid', drop_level=False)
    # adds to LL
    vitalsP_LL.append(dfIter)
# check if linked list is working
vitalsP_LL.display()
print(vitalsP_LL.length())

# %% 


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

# method to print measurements
def print_icp_counts(countAlive, count20_25, count25_30, count30_):
    print("# of Alive Patients: ", countAlive)
    print("# spikes w ICP 20-25: ", count20_25)
    print("# spikes w ICP 25-30: ", count25_30)
    print("# spikes w ICP 30+: ", count30_)

# %% systemicmeanImpute

# def systemicmeanImpute(vitalsP_DF):

#     # Replace negative values with NaN
#     vitalsP_DF.loc[vitalsP_DF['systemicmean'] < 0, 'systemicmean'] = np.nan
#     # create imputation variable
#     vitalsP_DF['meanBP'] = 0
#     mno.matrix(vitalsP_DF, figsize=(20, 6)) # displays NaN's in df_vitalsP

#     fwd = 0
#     bwd = 0
#     fwdDist = 0
#     bwdDist = 0
#     nanRemove = 0
#     booleanBwd = True
#     booleanFwd = True
#     # (Pre-processing) imputation for vitalsP (set final 75 nan's)
#     tempNode = vitalsP_LL.head
#     impVitals = LL()
#     print("nans left", vitalsP_DF['systemicmean'].isna().sum())

#     while tempNode:
#         dt = tempNode.data
#         time = dt.index.get_level_values('Time').to_numpy()
#         timeMissingSysMean = time[dt['systemicmean'].isna()]

#         print(dt.index.get_level_values('systemicmean').tolist())
#         # All impute per patient
#         for spot in timeMissingSysMean:
#                 # reset boolean values
#                 booleanBwd = True
#                 booleanFwd = True
#             # if empty data frame, skip imputation
#                 if(time.size == 0 or timeMissingSysMean.size == 0):
#                     print("nothing to impute")
#                     continue
#                 # time index of 'spot' to impute missing value
#                 spot_ind = np.where(time == spot)[0][0]


#                 # set fwd/bwd dist and values, see direction of imputation
#                 # Check forward imputation possibility
#                 if spot_ind + 1 >= len(time):
#                     booleanFwd = False
#                 else:
#                     fwd = time[spot_ind + 1]
#                     fwdDist = fwd - time[spot_ind]
#                 # Check backward imputation possibility
#                 if spot_ind - 1 < 0:
#                     booleanBwd = False
#                 else:
#                     bwd = time[spot_ind - 1]
#                     bwdDist = time[spot_ind] - bwd


#                 if(bwdDist > 5):
#                     booleanBwd = False
#                 if(fwdDist > 5):
#                     booleanFwd = False
#                 # conditions for imputation
#                 if not booleanBwd and not booleanFwd:
#                     print("can't impute")
#                 elif booleanFwd and not booleanBwd:
#                     dt.loc[dt.index.get_level_values('Time') == spot, 'systemicmean'] = dt.loc[dt.index.get_level_values('Time') == fwd, 'systemicmean']
#                     nanRemove += 1
#                 elif not booleanFwd and booleanBwd:
#                     dt.loc[dt.index.get_level_values('Time') == spot, 'systemicmean'] = dt.loc[dt.index.get_level_values('Time') == bwd, 'systemicmean']
#                     nanRemove += 1
#                 elif fwdDist < bwdDist:
#                     dt.loc[dt.index.get_level_values('Time') == spot, 'systemicmean'] = dt.loc[dt.index.get_level_values('Time') == fwd, 'systemicmean']
#                     nanRemove += 1
#                 elif fwdDist > bwdDist:
#                     dt.loc[dt.index.get_level_values('Time') == spot, 'systemicmean'] = dt.loc[dt.index.get_level_values('Time') == bwd, 'systemicmean']
#                     nanRemove += 1
#                 elif fwdDist == bwdDist:
#                     dt.loc[dt.index.get_level_values('Time') == spot, 'systemicmean'] = dt.loc[dt.index.get_level_values('Time') == fwd, 'systemicmean']
#                     nanRemove += 1

#         print(nanRemove)
#         nanRemove = 0
            

#         tempNode = tempNode.next

#     impVitals.append(dt)
# %% (imputation)

# systemicmeanImpute(vitalsP_DF)
# systemicsystoic
# heartrate
# respiration
# spo2
# temperature

# %% (Visualization) streamlit graphs

#                           ---------- All ICP v Time ----------
st.title('All ICP values vs Time')

fig = go.Figure()
for patient_id in vitalsP_DF['patientunitstayid'].unique():
    patient_data = vitalsP_DF[vitalsP_DF['patientunitstayid'] == patient_id]
    fig.add_trace(go.Scatter(x=patient_data['Time'], y=patient_data['icp'], mode='lines', name=f'Patient {patient_id}'))
fig.update_layout(title='All ICP values vs Time', xaxis_title='Time', yaxis_title='ICP Value', hovermode='closest')
fig.update_yaxes(range=[0, 50])

fig.show()

#               ---------- ICP of Alive/Expired patients, day 1-3 histograms ----------

# setting variables for following method
count20_25, count25_30, count30_, count25_30, count30_ = 0, 0, 0, 0, 0

within_24h, within_48h, within_72h = aliveID_DF[aliveID_DF['Time'] <= 24], aliveID_DF[aliveID_DF['Time'] <= 48], aliveID_DF[aliveID_DF['Time'] <= 72]
listAliveWithin = [within_24h, within_48h, within_72h]
expired_within_24h, expired_within_48h, expired_within_72h = expiredID_DF[expiredID_DF['Time'] <= 24], expiredID_DF[expiredID_DF['Time'] <= 48], expiredID_DF[expiredID_DF['Time'] <= 72]
listExpWithin = [expired_within_24h, expired_within_48h, expired_within_72h]
listChoose = [listAliveWithin, listExpWithin, aliveID_DF, expiredID_DF]

# printing the counts
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
vitalsP_DF_patientIDs = vitalsP_DF_patientIDs # unique patient id list 
vitalsP_LL = vitalsP_LL # vitalsP linked list 
vitalsP_MultiIndex = vitalsP_MultiIndex # vitalsP multi index 


# %% 

'''
     (\          
    (  \  /(o)\       
    (   \/  ()/ /)      Everything underneath here represents 
     (   `;.))'".)          the new code for the final deliverable. 
      `(/////.-'
   =====))=))===()
     ///'      
    //   PjP/ejm
   '    
'''

