# working on the final levels of visualization

# %%
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.linear_model import LinearRegression
import missingno as mno
from sklearn.preprocessing import MinMaxScaler
from statsmodels.imputation.mice import MICEData

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

df_vitalsP = pd.read_csv('vitalsP.csv')


icp_list = df_vitalsP['icp']
time_list = df_vitalsP['Time']
df_vitalsP.head()

df_vitalsP = df_vitalsP.drop(columns=['Unnamed: 0', 'observationoffset', 'Day', 'Hour', 'systemicdiastolic', 'systemicsystolic'])
# patient 1082792 literally has only one data point :) 
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

df_vitalsP = df_vitalsP.loc[df_vitalsP['patientunitstayid'].isin(patient_list)]
df_vitalsP.head()

# %% 
# scatterplot of icp vs time 

plt.figure() 
plt.figure(figsize=(10,6))
sns.lineplot(data = df_vitalsP, x = time_list, y = icp_list, palette='bright', hue = 'patientunitstayid', legend=False)
plt.ylim(0, 50)




 # %%

mno.matrix(df_vitalsP, figsize=(20, 6))

# %%
# median of the last 30 minutes in ICP 
# df_vitalsP = df_vitalsP.fillna(df_vitalsP.median())
# df_apache = df_apache.dropna()
# df_infs = df_infs.dropna()
print(df_vitalsP.dtypes)


#%%
df_vitalsP = df_vitalsP.fillna(method='ffill')


#%%
df_vitalsP.head()

#%%
mno.matrix(df_vitalsP, figsize=(20, 6))

#%%
#testing for data correlation between all variables
missing_corr = df_vitalsP.isnull().corr()
print(missing_corr)


# %%
mno.matrix(df_vitalsP, figsize=(20, 6))

# %%

# Now we make the giant linked list structure for um everything
# df_vitalsP = df_vitalsP.sort_values(by=['patientunitstayid', 'Time'])


df_vitalsP = df_vitalsP.sort_values(by=['patientunitstayid', 'Time'])

df_vitalCopy = df_vitalsP

df_vitalsP = df_vitalsP.set_index(['patientunitstayid', 'Time'])


# %%

# check that it worked 
# df_vitalsP.loc[193629]

# %%

unique_patient_ids = df_vitalsP.index.get_level_values('patientunitstayid').unique()
print(unique_patient_ids)
print(len(unique_patient_ids))


orig_set = set(patient_list)
gen_set = set(unique_patient_ids)
missing_in_generated = orig_set - gen_set
print(f"missing ids from og list but not in gen list {missing_in_generated}")

# %% 

# with set_names we ensured that the dataframe has the correct index every time

dfL_vitals = LL()

for patient_id in unique_patient_ids: 
    # dfIter = df_vitalsP.loc[patient_id]
    # should get datframe for each patient
    dfIter = df_vitalsP.xs(patient_id, level='patientunitstayid', drop_level=False)
    # dfIter.index.set_names(['patientunitstayid', 'Time'], inplace=True)
    dfL_vitals.append(dfIter)

dfL_vitals.display()
print(dfL_vitals.length())



# %%

# This is the debugging print
# tempNode = dfL_vitals.head
# count = 0
# while tempNode: 
#     dt = tempNode.data
#     print("Current index:", dt.index)
#     print("Index names:", dt.index.names)
#     print(f'The patient count is {count}')
#     # print(dt.head())
#     patient = dt.index.get_level_values('patientunitstayid').unique()[0]
#     count += 1
#     tempNode = tempNode.next

# %%
# ----- Actual graphs are below here -----

# tempNode = dfL_vitals.head
# count = 0

# while tempNode:
#     print(f"The count is {count}")
#     dt = tempNode.data
#     print(dt.index.get_level_values('Time'))
    

#     # there was a bracket 0 at the end for some reason
#     patient = dt.index.get_level_values('patientunitstayid').unique()[0]
    
#     # Select only numeric columns
#     numeric_columns = dt.select_dtypes(include=[np.number]).columns
    
#     # Create a bigger figure to accommodate multiple plots
#     fig, ax = plt.subplots(figsize=(15, 10))
#     plt.title(f"Patient ID: {patient}", fontsize=16)
    
#     # Plot each numeric column
#     for column in numeric_columns:
#         if column != 'Time':  # Exclude 'Time' as it's our x-axis
#             sns.lineplot(data=dt, x='Time', y=column, label=column, ax=ax)

#     if(dt.index.get_level_values('Time').max() > 72):
#         plt.xlim(0, 72)
#     plt.xlabel('Time', fontsize=12)
#     plt.ylabel('Value', fontsize=12)
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()
#     plt.show()
    
    
#     count += 1
#     tempNode = tempNode.next

# %%
# ------------------- more important graphs -----------------------
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

df_expired = df_vitalCopy[df_vitalCopy['patientunitstayid'].isin(expired_list)]
df_alive = df_vitalCopy[df_vitalCopy['patientunitstayid'].isin(alive_list)]

# plt.figure(figsize=(14, 6))

# for patient_id in df_alive['patientunitstayid'].unique():
#     patient_data = df_alive[df_alive['patientunitstayid'] == patient_id]
#     # plt.plot(patient_data['Time'], patient_data['icp'], label=f'Patient {patient_id}')
#     sns.lineplot(data = patient_data, x = 'Time', y = 'icp')
# plt.ylim(5, 55)

# plt.title('ICP Values of Alive Patients')
# plt.xlabel('Time')
# plt.ylabel('ICP Value')

# # %%
# # Graph for the expired patients 

# plt.figure(figsize=(14, 6))

# for patient_id in df_expired['patientunitstayid'].unique():
#     patient_data = df_expired[df_expired['patientunitstayid'] == patient_id]
#     plt.plot(patient_data['Time'], patient_data['icp'], label=f'Patient {patient_id}')
# plt.ylim(5, 40)
# plt.title('ICP Values of Expired Patients')
# plt.xlabel('Time')
# plt.ylabel('ICP Value')


# %%


# ----------------------- ACTUAL STREAMLIT GRAPHS ------------------------------



# %%

st.title('All ICP values vs Time')
fig = go.Figure()
for patient_id in df_vitalCopy['patientunitstayid'].unique():
    patient_data = df_vitalCopy[df_vitalCopy['patientunitstayid'] == patient_id]
    fig.add_trace(go.Scatter(x=patient_data['Time'], y=patient_data['icp'], mode='lines', name=f'Patient {patient_id}'))

fig.update_layout(title='All ICP values vs Time', xaxis_title='Time', yaxis_title='ICP Value', hovermode='closest')
fig.update_yaxes(range=[0, 50])
st.plotly_chart(fig)


# %%
# Bell curve histogram 
with st.expander(f'Histogram of ICP Values for All Patients'):
    # Combine ICP values from all patients
    all_icp_values = df_vitalCopy['icp']

    # Create a histogram
    fig = go.Figure(data=[go.Histogram(x=all_icp_values, xbins=dict(start=0, end=50, size=1))])

    fig.update_layout(title='Histogram of ICP Values for All Patients',
                    xaxis_title='ICP Value',
                    yaxis_title='Count',
                    bargap=0.2)

    st.plotly_chart(fig)

# %%

st.title('Interactive ICP of Alive patients')


fig = go.Figure()

for patient_id in df_alive['patientunitstayid'].unique():
    patient_data = df_alive[df_alive['patientunitstayid'] == patient_id]
    fig.add_trace(go.Scatter(x=patient_data['Time'], y=patient_data['icp'], mode='lines', name=f'Patient {patient_id}'))

fig.update_layout(title='ICP Values of Alive Patients', xaxis_title='Time', yaxis_title='ICP Value', hovermode='closest')
fig.update_yaxes(range=[5, 55])

st.plotly_chart(fig)


with st.expander(f'Histogram of ICP Values for Alive Patients'):
    # Combine ICP values from all patients
    all_icp_values = df_alive['icp']

    # Create a histogram
    fig = go.Figure(data=[go.Histogram(x=all_icp_values, xbins=dict(start=0, end=50, size=1))])

    fig.update_layout(title='Histogram of ICP Values for Alive Patients',
                    xaxis_title='ICP Value',
                    yaxis_title='Count',
                    bargap=0.2)

    st.plotly_chart(fig)
    
    multi = ''' We noticed a :blue-background[normal, right skewed distribution curve] as expected for alive patients.  
    '''

    st.markdown(multi)
# -------

st.title('Interactive ICP of Expired Patients')

fig = go.Figure()

for patient_id in df_expired['patientunitstayid'].unique():
    patient_data = df_expired[df_expired['patientunitstayid'] == patient_id]
    fig.add_trace(go.Scatter(x=patient_data['Time'], y=patient_data['icp'], mode='lines', name=f'Patient {patient_id}'))

fig.update_layout(title='Interactive ICP of Expired Patients', xaxis_title='Time', yaxis_title='ICP Value', hovermode='closest')
fig.update_yaxes(range=[5, 55])

st.plotly_chart(fig)


with st.expander(f'Histogram of ICP Values for Expired Patients'):
    # Combine ICP values from all patients
    all_icp_values = df_expired['icp']

    # Create a histogram
    fig = go.Figure(data=[go.Histogram(x=all_icp_values, xbins=dict(start=0, end=50, size=1))])

    fig.update_layout(title='Histogram of ICP Values for Expired Patients',
                    xaxis_title='ICP Value',
                    yaxis_title='Count',
                    bargap=0.2)

    st.plotly_chart(fig)

    multi = ''' However, for expired patients we noticed :blue-background[survivorship bias]. We would have to split by time.  
    We also do note that the mean for this data is to the right of the previous graph for higher ICP values on average. 
    '''

    st.markdown(multi)

# %%

st.title('Vitals of Every Patient')

tempNode = dfL_vitals.head
while tempNode: 
    dt = tempNode.data
    patient = dt.index.get_level_values('patientunitstayid').unique()[0]
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

        st.plotly_chart(fig)
    
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

#%%
#Here we estimate the area under. Start by excluding irrelevant values
# df_vitalsP = df_vitalsP.reset_index()
# df_vitalsP['icpSpike20to25'] = df_vitalsP['icp'].where((df_vitalsP['icp'] >= 20) & (df_vitalsP['icp'] <= 25))
# df_vitalsP['icpSpike25to30'] = df_vitalsP['icp'].where((df_vitalsP['icp'] >= 25) & (df_vitalsP['icp'] <= 30))
# df_vitalsP['icpSpike30to35'] = df_vitalsP['icp'].where((df_vitalsP['icp'] >= 30) & (df_vitalsP['icp'] <= 35))
# df_vitalsP['icpSpike35+'] = df_vitalsP['icp'].where(df_vitalsP['icp'] >= 35)
# %%

#%%
df_vitalsP.head()



# %%
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
maximum_icp = df_vitalCopy['icp'].max()
# before we added (35,maximum_icp)
icp_ranges = [(0, 20), (20, 25), (25, 30), (30, 35), (35, maximum_icp)]
# Using linked list code to make our lives easier 

def func_icp_range(DF):
    df_icp_ranges = LL()
    for df_value in icp_ranges: 

        min_icp = df_value[0]
        max_icp = df_value[1]  
        
        # Filter corresponding time column based on the same condition
        filtered_df = DF.loc[(DF['icp'] >= min_icp) & (DF['icp'] <= max_icp), ['icp', 'Time']]

        # Create a new DataFrame with filtered data
        df_icp_ranges.append(filtered_df)
    return df_icp_ranges

df_vitalCopySORTED = df_vitalCopy.sort_values(by=['Time'])

df_icp_ranges = func_icp_range(df_vitalCopy)
df_icp_ranges.display()
print(df_icp_ranges.length())


# %%

def range_traversal(df_icp_ranges):
    tempNode = df_icp_ranges.head
    count = 0
    sumTotal = 0
    while tempNode: 
        dt = tempNode.data
        
        freq = dt['icp'].sum()
        
        range_check = icp_ranges[count]
        
        # y should be first for calculating area under the 
        # curve. trapezoidal riemann
        ipc_load = np.trapz(dt['icp'], dt['Time'])
        sumTotal += ipc_load
        
        print(f"For range {range_check }, frequency is {freq} and ipc_load is {ipc_load}")

        count += 1
        tempNode = tempNode.next
    print(f"THE ACTUAL TOTALED SUM IS {sumTotal}")

range_traversal(df_icp_ranges)

# %%

# use both functions for each patient! 

tempNode = dfL_vitals.head

while tempNode: 
    dt = tempNode.data
    patient = dt.index.get_level_values('patientunitstayid').unique()[0]
    # time = dt.index.get_level_values('Time')

    dt = dt.reset_index()
    # total_ipc_area = np.trapz(dt['icp'], dt['Time'])    

    # print(f'The total ipc area for this patient is {total_ipc_area}')
    dt_linked_list = func_icp_range(dt)

    # This is our print function 
    print(f"For patient {patient}")
    range_traversal(dt_linked_list)
    print("\n")

    tempNode = tempNode.next




# %%

'''
struggling parrots
     _
    /")   @, 
   //)   /)
==/{'==='%"===
 /'     %
        %              ejm   
         %
'''



# %%


'''
            icp load for the first 7 days, if available
| patient |  24 hr | 48 hr | .. 

def : 
    24 hour trapz 
    48 hour trapz 

'''


time_ranges = [(0, 24), (24, 48), (48, 72), (72, 96), (96, 120), (120, 144), (144, 168)]
# Using linked list code to make our lives easier 

def day_icp_load(patient_df, patient):
    df_time_ranges = LL()
    df_icp_loads = LL()
    for df_value in time_ranges: 

        min_time = df_value[0]
        max_time = df_value[1]  
        
        # Filter corresponding time column based on the same condition
        df_day = patient_df.loc[(patient_df['Time'] >= min_time) & (patient_df['Time'] <= max_time), ['icp', 'Time']]     
        
        # plot icp against time 
        plt.figure()
        sns.lineplot(data = df_day, x = df_day['Time'], y = df_day['icp'])
        plt.title(f'Patient {patient}')

        # Create a new DataFrame with filtered data
        df_time_ranges.append(df_day)
    
    # now use df_time_ranges

    # icp_load = 0

    # tempNode = df_time_ranges.head
    # while tempNode:
    #     dt = tempNode.data
    #     icp_load = np.trapz(dt['Time'], dt['icp'])
    #     # append to the actual linked list 
    #     df_icp_loads.append(icp_load)
    #     tempNode = tempNode.next

    # return df_icp_loads


# now traverse through linked list of patients and calcualte linked list of icp loads for each

DF_DAYS = pd.DataFrame(columns=['patientunitstayid'])

tempNode = dfL_vitals.head
while tempNode: 
    dt = tempNode.data
    patient = dt.index.get_level_values('patientunitstayid').unique()[0]

    DF_DAYS.loc[len(DF_DAYS), 'patientunitstayid'] = patient
    # DF_DAYS['patientunitstayid'] = patient
    time = dt.index.get_level_values('Time')
    
    dt = dt.reset_index()

    # icp load list, then iterate through the linked list, adding each as its own column
    print(f'Patient {patient}')
    icp_load_list = LL()
    icp_load_list = day_icp_load(dt, patient)

    break
    # tempNode_icp = icp_load_list.head
    count = 1


    # while tempNode_icp:
    #     DF_DAYS.loc[len(DF_DAYS), f'Day {count}'] = tempNode_icp.data
    #     # DF_DAYS[f'Day {count}'] = tempNode_icp.data
    #     tempNode_icp = tempNode_icp.next
    #     count += 1
    
    tempNode = tempNode.next


# DF_DAYS.head()

    

# %%


'''
hopefully successful parrots
     _
    /")   @, 
   //)   /)
==/{'==='%"===
 /'     %
        %              ejm   
         %
'''