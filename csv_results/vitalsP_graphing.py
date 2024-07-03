# working on the final levels of visualization





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

df_apache = pd.read_csv('apache_results.csv')
df_diagP = pd.read_csv('diagP_results.csv')
df_infs = pd.read_csv('infsP_results.csv')
df_labsP = pd.read_csv('labsP_results.csv')
df_examP = pd.read_csv('examP_results.csv')
# %% (Pre-processing) dropping columns from df_vitalsP
df_vitalsP = pd.read_csv('vitalsP.csv')

df_vitalsP = df_vitalsP.drop(columns=['Unnamed: 0', 'observationoffset', 'Day', 'Hour', 'systemicdiastolic', 'systemicsystolic'])
#%% (FIX)(Pre-processing) imputation for vitalsP
mno.matrix(df_vitalsP, figsize=(20, 6)) # displays NaN's in df_vitalsP
# Step 1: Separate the column you want to exclude
systemicMean_column = df_vitalsP['systemicmean'].copy()

# Step 2: Perform the forward fill on the modified DataFrame
df_vitalsP.fillna(method='ffill', inplace=True)

# Step 3: Reintegrate the excluded column
df_vitalsP['systemicmean'] = systemicMean_column

# %% (Pre-processing) multi-index vitalsP on id and time

df_vitalsP = df_vitalsP.sort_values(by=['patientunitstayid', 'Time'])
df_vitalCopy = df_vitalsP.copy() #Deep copy of df_vitalsP

df_vitalsP = df_vitalsP.set_index(['patientunitstayid', 'Time']) #Setting the multiindex by patient_id and time

display(df_vitalsP) #check if multiindex worked 

# %% (Pre-processing) ensure vitalsP has correct patient ids

# patient 1082792 has only a single data point
# missing ids from og list but not in gen list {2782239}
patient_list = [
    306989, 1130290, 1210520, 1555058, 1580984, 2375786, 2823473,
    2887235, 2898120, 3075566, 3336597, 193629, 263556, 272638, 621883, 799478, 1079428, 1082792,
    1088266, 1092809, 1116007, 1162658, 1175888, 1535342, 1556670, 2198292,
    2247037, 2405050, 2671145, 2683425, 2689775, 2721908, 2722053, 2724565,
    2725853, 2767039, 2768739, 2773734, 2803129, 2846229, 2870532,
    2885054, 2890935, 2895083, 3064120, 3100062, 3210988, 3212405, 3214569,
    3217832, 3222024, 3245093, 3347750, 2782239 
]

#retrieves patient ids from the now multiindexed df_vitalsP
unique_patient_ids = df_vitalsP.index.get_level_values('patientunitstayid').unique()

#find the missing id's
orig_set = set(patient_list)
gen_set = set(unique_patient_ids)
missing_in_generated = orig_set - gen_set
print(f"missing ids from og list but not in gen list {missing_in_generated}")

# df_vitalsP now only retains the patient ids that are in patient_list
df_vitalsP = df_vitalsP.loc[df_vitalsP.index.get_level_values('patientunitstayid').isin(patient_list)]
# %% (Pre-Visualization) Creating Linked List of vitalsP
print("hello")
dfL_vitals = LL() #LL object

# Each unique id is used to identify multi index objects
for patient_id in unique_patient_ids: 
    # gets a multi index object of a specific id, appends to LL
    df_multiobj = df_vitalsP[df_vitalsP.index.get_level_values('patientunitstayid') == patient_id]
    dfL_vitals.append(df_multiobj)

dfL_vitals.display()
print(dfL_vitals.length())


# %% (Pre-Visualization) Prepare for graphing

# note that for patient 1210520, there is a discrepancy between the two values 
# we sure that we're ok with that? 

expired_list = [306989, 1130290, 1210520, 1555058, 1580984, 2375786, 2823473,
    2887235, 2898120, 3075566, 3336597]

alive_list = [193629, 263556, 272638, 621883, 799478, 1079428, 1082792,
    1088266, 1092809, 1116007, 1162658, 1175888, 1535342, 1556670, 2198292,
    2247037, 2405050, 2671145, 2683425, 2689775, 2721908, 2722053, 2724565,
    2725853, 2767039, 2768739, 2773734, 2803129, 2846229, 2870532,
    2885054, 2890935, 2895083, 3064120, 3100062, 3210988, 3212405, 3214569,
    3217832, 3222024, 3245093, 3347750, 2782239]

# Expired and Alive patient id's
df_expired = df_vitalCopy[df_vitalCopy['patientunitstayid'].isin(expired_list)]
df_alive = df_vitalCopy[df_vitalCopy['patientunitstayid'].isin(alive_list)]

# method to print measurements
def print_icp_counts(countAlive, count20_25, count25_30, count30_):
    print("# of Alive Patients: ", countAlive)
    print("# spikes w ICP 20-25: ", count20_25)
    print("# spikes w ICP 25-30: ", count25_30)
    print("# spikes w ICP 30+: ", count30_)


# %% (Visualization) STREAMLIT GRAPHS

# ----------------------- ACTUAL STREAMLIT GRAPHS ------------------------------



# %% All ICP v Time

st.title('All ICP values vs Time')
fig = go.Figure()
for patient_id in df_vitalCopy['patientunitstayid'].unique():
    patient_data = df_vitalCopy[df_vitalCopy['patientunitstayid'] == patient_id]
    fig.add_trace(go.Scatter(x=patient_data['Time'], y=patient_data['icp'], mode='lines', name=f'Patient {patient_id}'))

fig.update_layout(title='All ICP values vs Time', xaxis_title='Time', yaxis_title='ICP Value', hovermode='closest')
fig.update_yaxes(range=[0, 50])
fig.show()




# %% Interactive ICP of Alive and expired patients, day 1-3 histogram

st.title('Interactive ICP of Alive patients')


fig = go.Figure()

for patient_id in df_alive['patientunitstayid'].unique():
    patient_data = df_alive[df_alive['patientunitstayid'] == patient_id]
    fig.add_trace(go.Scatter(x=patient_data['Time'], y=patient_data['icp'], mode='lines', name=f'Patient {patient_id}'))

fig.update_layout(title='ICP Values of Alive Patients', xaxis_title='Time', yaxis_title='ICP Value', hovermode='closest')
fig.update_yaxes(range=[5, 55])

fig.show()

count25_30 = df_alive[(df_alive['icp'] >= 25) & (df_alive['icp'] <= 30)].shape[0]
count30_ = df_alive[(df_alive['icp'] >= 30)].shape[0]
count20_25 = df_alive[(df_alive['icp'] >= 20) & (df_alive['icp'] <= 25)].shape[0]
countAlive = df_alive['patientunitstayid'].unique().shape[0]

print_icp_counts(countAlive, count20_25, count25_30, count30_)

# Bell curve histogram 
with st.expander(f'Day 1 ICP Histogram for Alive Patients'):
    # Filter to make a dataframe with only rows from the first 24 hours
    within_24h = df_alive[df_alive['Time'] <= 24]

    # Select only the 'icp' column values from the new dataframe
    icp_values_within_24h = within_24h['icp']

    # Create a histogram
    fig = go.Figure(data=[go.Histogram(x=icp_values_within_24h, xbins=dict(start=0, end=50, size=1))])

    fig.update_layout(title='Day 1 ICP Histogram for Alive Patients',
                      xaxis_title='ICP Value',
                      yaxis_title='Count',
                      bargap=0.2)
    
    fig.show()

    count25_30 = within_24h[(within_24h['icp'] >= 25) & (within_24h['icp'] <= 30)].shape[0]
    count30_ = within_24h[(within_24h['icp'] >= 30)].shape[0]
    count20_25 = within_24h[(within_24h['icp'] >= 20) & (within_24h['icp'] <= 25)].shape[0]
    countAlive = within_24h['patientunitstayid'].unique().shape[0]
    
    print_icp_counts(countAlive, count20_25, count25_30, count30_)


with st.expander(f'Day 2 ICP Histogram for Alive Patients'):
    # Corrected filtering condition with proper parentheses
    within_48h = df_alive[(df_alive['Time'] >= 24) & (df_alive['Time'] <= 48)]

    # Select only the 'icp' column values from the new dataframe
    icp_values_within_48h = within_48h['icp']

    # Corrected the variable used for plotting the histogram
    fig = go.Figure(data=[go.Histogram(x=icp_values_within_48h, xbins=dict(start=0, end=50, size=1))])

    fig.update_layout(title='Day 2 ICP Histogram for Alive Patients',
                      xaxis_title='ICP Value',
                      yaxis_title='Count',
                      bargap=0.2)

    fig.show()

    count25_30 = within_48h[(within_48h['icp'] >= 25) & (within_48h['icp'] <= 30)].shape[0]
    count30_ = within_48h[(within_48h['icp'] >= 30)].shape[0]
    count20_25 = within_48h[(within_48h['icp'] >= 20) & (within_48h['icp'] <= 25)].shape[0]
    countAlive = within_48h['patientunitstayid'].unique().shape[0]
    
    print_icp_counts(countAlive, count20_25, count25_30, count30_)


with st.expander(f'Day 3 ICP Histogram for Alive Patients'):
    # Corrected filtering condition with proper parentheses
    within_72h = df_alive[(df_alive['Time'] >= 48) & (df_alive['Time'] <= 72)]

    # Select only the 'icp' column values from the new dataframe
    icp_values_within_72h = within_72h['icp']

    # Corrected the variable used for plotting the histogram
    fig = go.Figure(data=[go.Histogram(x=icp_values_within_72h, xbins=dict(start=0, end=50, size=1))])

    fig.update_layout(title='Day 3 ICP Histogram for Alive Patients',
                      xaxis_title='ICP Value',
                      yaxis_title='Count',
                      bargap=0.2)

    fig.show()

    count25_30 = within_72h[(within_72h['icp'] >= 25) & (within_72h['icp'] <= 30)].shape[0]
    count30_ = within_72h[(within_72h['icp'] >= 30)].shape[0]
    count20_25 = within_72h[(within_72h['icp'] >= 20) & (within_72h['icp'] <= 25)].shape[0]
    countAlive = within_72h['patientunitstayid'].unique().shape[0]
    
    print_icp_counts(countAlive, count20_25, count25_30, count30_)


# Corrected code for filtering dataframes and using the correct variable for histogram data

# Bell curve histogram

st.title('Interactive ICP of Expired patients')


fig = go.Figure()

for patient_id in df_expired['patientunitstayid'].unique():
    patient_data = df_expired[df_expired['patientunitstayid'] == patient_id]
    fig.add_trace(go.Scatter(x=patient_data['Time'], y=patient_data['icp'], mode='lines', name=f'Patient {patient_id}'))

fig.update_layout(title='ICP Values of Expired Patients', xaxis_title='Time', yaxis_title='ICP Value', hovermode='closest')
fig.update_yaxes(range=[5, 55])

fig.show()

count25_30 = df_expired[(df_expired['icp'] >= 25) & (df_expired['icp'] <= 30)].shape[0]
count30_ = df_expired[(df_expired['icp'] >= 30)].shape[0]
count20_25 = df_expired[(df_expired['icp'] >= 20) & (df_expired['icp'] <= 25)].shape[0]
countAlive = df_expired['patientunitstayid'].unique().shape[0]

print_icp_counts(countAlive, count20_25, count25_30, count30_)


with st.expander(f'Day 1 ICP Histogram for Expired Patients'):
    # Filter to make a dataframe with only rows from the first 24 hours
    expired_within_24h = df_expired[df_expired['Time'] <= 24]

    # Select only the 'icp' column values from the new dataframe
    icp_values_within_24h = expired_within_24h['icp']

    # Create a histogram
    fig = go.Figure(data=[go.Histogram(x=icp_values_within_24h, xbins=dict(start=0, end=50, size=1))])

    fig.update_layout(title='Day 1 ICP Histogram for Expired Patients',
                      xaxis_title='ICP Value',
                      yaxis_title='Count',
                      bargap=0.2)

    fig.show()
    count25_30 = expired_within_24h[(expired_within_24h['icp'] >= 25) & (expired_within_24h['icp'] <= 30)].shape[0]
    count30_ = expired_within_24h[(expired_within_24h['icp'] >= 30)].shape[0]
    count20_25 = expired_within_24h[(expired_within_24h['icp'] >= 20) & (expired_within_24h['icp'] <= 25)].shape[0]
    countExpired = expired_within_24h['patientunitstayid'].unique().shape[0]
    
    print_icp_counts(countAlive, count20_25, count25_30, count30_)


with st.expander(f'Day 2 ICP Histogram for Expired Patients'):
    # Corrected filtering condition with proper parentheses
    expired_within_48h = df_expired[(df_expired['Time'] >= 24) & (df_expired['Time'] <= 48)]

    # Select only the 'icp' column values from the new dataframe
    icp_values_within_48h = expired_within_48h['icp']

    # Corrected the variable used for plotting the histogram
    fig = go.Figure(data=[go.Histogram(x=icp_values_within_48h, xbins=dict(start=0, end=50, size=1))])

    fig.update_layout(title='Day 2 ICP Histogram for Expired Patients',
                      xaxis_title='ICP Value',
                      yaxis_title='Count',
                      bargap=0.2)

    fig.show()

    count25_30 = expired_within_48h[(expired_within_48h['icp'] >= 25) & (expired_within_48h['icp'] <= 30)].shape[0]
    count30_ = expired_within_48h[(expired_within_48h['icp'] >= 30)].shape[0]
    count20_25 = expired_within_48h[(expired_within_48h['icp'] >= 20) & (expired_within_48h['icp'] <= 25)].shape[0]
    countExpired = expired_within_48h['patientunitstayid'].unique().shape[0]
    
    print_icp_counts(countAlive, count20_25, count25_30, count30_)


with st.expander(f'Day 3 ICP Histogram for Expired Patients'):
    # Corrected filtering condition with proper parentheses
    expired_within_72h = df_expired[(df_expired['Time'] >= 48) & (df_expired['Time'] <= 72)]

    # Select only the 'icp' column values from the new dataframe
    icp_values_within_72h = expired_within_72h['icp']

    # Corrected the variable used for plotting the histogram
    fig = go.Figure(data=[go.Histogram(x=icp_values_within_72h, xbins=dict(start=0, end=50, size=1))])

    fig.update_layout(title='Day 3 ICP Histogram for Expired Patients',
                      xaxis_title='ICP Value',
                      yaxis_title='Count',
                      bargap=0.2)

    fig.show()

    count25_30 = expired_within_72h[(expired_within_72h['icp'] >= 25) & (expired_within_72h['icp'] <= 30)].shape[0]
    count30_ = expired_within_72h[(expired_within_72h['icp'] >= 30)].shape[0]
    count20_25 = expired_within_72h[(expired_within_72h['icp'] >= 20) & (expired_within_72h['icp'] <= 25)].shape[0]
    countExpired = expired_within_72h['patientunitstayid'].unique().shape[0]
    
    print_icp_counts(countAlive, count20_25, count25_30, count30_)

# %% Vitals of each patient with ICP
st.title('Vitals of Every Patient')

tempNode = dfL_vitals.head
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








# ----------------very big line break ----------------------

#%% (prep code) AUC for ICP spikes
#Here we estimate the area under. Start by excluding irrelevant values
df_vitalsP = df_vitalsP.reset_index()
df_vitalsP['icpSpike20to25'] = df_vitalsP['icp'].where((df_vitalsP['icp'] >= 20) & (df_vitalsP['icp'] <= 25))
df_vitalsP['icpSpike25to30'] = df_vitalsP['icp'].where((df_vitalsP['icp'] >= 25) & (df_vitalsP['icp'] <= 30))
df_vitalsP['icpSpike30to35'] = df_vitalsP['icp'].where((df_vitalsP['icp'] >= 30) & (df_vitalsP['icp'] <= 35))
df_vitalsP['icpSpike35+'] = df_vitalsP['icp'].where(df_vitalsP['icp'] >= 30)

# %% 

# method for area, plotting data points
def process_icp_range(df_vitalsP, time_col, icp_col, min_icp, max_icp):
    # Create ICP range column
    range_col = f'icpSpike{min_icp}to{max_icp}'
    df_vitalsP[range_col] = df_vitalsP[icp_col].where((df_vitalsP[icp_col] >= min_icp) & (df_vitalsP[icp_col] <= max_icp))
    
    # Prepare data
    df_clean = df_vitalsP.reset_index()
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
    total_points = len(df_vitalsP)
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
    area, clean_points, total_points = process_icp_range(df_vitalsP, 'Time', 'icp', min_icp, max_icp)
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
# %%
# %% part b(systemicmean)

# Replace negative values with NaN
df_vitalsP.loc[df_vitalsP['systemicmean'] < 0, 'systemicmean'] = np.nan


df_vitalsP['meanBP'] = 0
mno.matrix(df_vitalsP, figsize=(20, 6)) # displays NaN's in df_vitalsP

#in hours

countNot = 0
nanRemove = 0
bwd_Dist = 0
bwd = 0
fwd_Dist = 0
fwd = 0
print("nans left", df_vitalsP['systemicmean'].isna().sum())
# Impute 
tempNode = dfL_vitals.head
while tempNode: 
    tempNode = tempNode.next
    dt = tempNode.data
    # create time list, and missing time list
    time = dt.index.get_level_values('Time').to_numpy()
    timeMissingSysMean = time[dt['systemicmean'].isna()]
    # All impute per patient
    for spot in timeMissingSysMean:
            # nothing to impute, skip
            if(time.size == 0 or timeMissingSysMean.size == 0):
                continue
            # find index of spot to impute
            spot_ind = np.where(time == spot)[0][0]
            # if last index, default impute bwd
            if(spot_ind+1 >= len(time)):
                fwd = [spot_ind]
                fwd_Dist = 6
            else:
                fwd = time[spot_ind+1]
                fwdDist = fwd-spot
            if(spot_ind-1 < 0):
                bwd = time[spot_ind]
                bwd_Dist = 6
            else:
                bwd = time[spot_ind-1]
                bwdDist = spot-bwd

            if(spot_ind+1 >= len(time) and spot_ind-1 < 0):
                print("both")
                print("fwd", fwd_Dist)
                print("bwd", bwd_Dist)
                print(" ")
            if(spot_ind+1 < len(time) and spot_ind-1 < 0):
                print("error bwd")
                print("fwd", fwd_Dist)
                print("bwd", bwd_Dist)
                print(" ")
            if(spot_ind+1 >= len(time) and spot_ind-1 >= 0):
                print("error fwd")
                print("fwd", fwd_Dist)
                print("bwd", bwd_Dist)
                print("")
                
            # go to next index value, if inside 5 hr, save the index pos
            # fwd beyond 5 hours
            if( (bwdDist)>5  and (fwdDist>5) ):
                print("can't impute")
                countNot+=1
            elif(bwdDist>5 and fwd_Dist<5): # impute fwd ..................
                print(2)
                try:
                    dt.loc[dt.index.get_level_values('Time') == spot, 
                        'systemicmean'] = dt.loc[dt.index.get_level_values('Time') == fwd, 'systemicmean']
                except:
                    print("fwd", fwd)
                nanRemove+=1
            elif(fwdDist>5 and bwd_Dist<5): # impute bwd ..................
                print(2)
                try:
                    dt.loc[dt.index.get_level_values('Time') == spot, 
                        'systemicmean'] = dt.loc[dt.index.get_level_values('Time') == bwd, 'systemicmean']
                except:
                    print("bwd 2", bwd)
                nanRemove+=1
            elif(fwdDist<bwdDist): # impute fwd
                try:
                    dt.loc[dt.index.get_level_values('Time') == spot, 
                        'systemicmean'] = dt.loc[dt.index.get_level_values('Time') == fwd, 'systemicmean']
                except:
                    print("fwd", fwd)
                
                nanRemove+=1
            elif(fwdDist>bwdDist): # impute bwd 
                # dt.loc[dt.index.get_level_values('Time') == spot, 
                #     'systemicmean'] = dt.loc[dt.index.get_level_values('Time') == bwd, 'systemicmean'].values[0]
                try:
                    dt.loc[dt.index.get_level_values('Time') == spot, 
                        'systemicmean'] = dt.loc[dt.index.get_level_values('Time') == bwd, 'systemicmean']
                except:
                    print("bwd 4", bwd)
                    print("fdist, bdist", bwdDist, fwdDist)
                    print("index bwd", dt.loc[dt.index.get_level_values('Time') == bwd])
                    print("index spot", dt.loc[dt.index.get_level_values('Time') == spot])
                    print(" ")
                    print(dt)
                    print(" ")
                nanRemove+=1
        
            bwd_Dist = 0
            bwd = 0
            fwd_Dist = 0
            fwd = 0
    tempNode = tempNode.next

print("nans left", df_vitalsP['systemicmean'].isna().sum())
print("nans should've removed", nanRemove)

mno.matrix(df_vitalsP, figsize=(20, 6)) # displays NaN's in df_vitalsP

