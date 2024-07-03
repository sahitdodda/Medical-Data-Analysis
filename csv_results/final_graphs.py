# working on the final levels of visualization

# %%

df_apache = pd.read_csv('apache_results.csv')
df_diagP = pd.read_csv('diagP_results.csv')
df_infs = pd.read_csv('infsP_results.csv')
df_labsP = pd.read_csv('labsP_results.csv')
df_examP = pd.read_csv('examP_results.csv')

df_vitalsP = pd.read_csv('vitalsP.csv')
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


import sys 
sys.path.append('LinkedListClass.py')
from LinkedListClass import Node, LL


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
tempNode = dfL_vitals.head
count = 0
while tempNode: 
    dt = tempNode.data
    print("Current index:", dt.index)
    print("Index names:", dt.index.names)
    print(f'The patient count is {count}')
    # print(dt.head())
    patient = dt.index.get_level_values('patientunitstayid').unique()[0]
    count += 1
    tempNode = tempNode.next

# %%
# ----- Actual graphs are below here -----
tempNode = dfL_vitals.head
count = 0

while tempNode:
    print(f"The count is {count}")
    dt = tempNode.data
    print(dt.head())
    print(dt.index.get_level_values('Time'))
    

    # there was a bracket 0 at the end for some reason
    patient = dt.index.get_level_values('patientunitstayid').unique()[0]
    
    # Select only numeric columns
    numeric_columns = dt.select_dtypes(include=[np.number]).columns
    
    # Create a bigger figure to accommodate multiple plots
    fig, ax = plt.subplots(figsize=(15, 10))
    plt.title(f"Patient ID: {patient}", fontsize=16)
    
    # Plot each numeric column
    for column in numeric_columns:
        if column != 'Time':  # Exclude 'Time' as it's our x-axis
            sns.lineplot(data=dt, x='Time', y=column, label=column, ax=ax)

    if(dt.index.get_level_values('Time').max() > 72):
        plt.xlim(0, 72)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    
    count += 1
    tempNode = tempNode.next

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

plt.figure(figsize=(14, 6))

for patient_id in df_alive['patientunitstayid'].unique():
    patient_data = df_alive[df_alive['patientunitstayid'] == patient_id]
    # plt.plot(patient_data['Time'], patient_data['icp'], label=f'Patient {patient_id}')
    sns.lineplot(data = patient_data, x = 'Time', y = 'icp')
plt.ylim(5, 55)

plt.title('ICP Values of Alive Patients')
plt.xlabel('Time')
plt.ylabel('ICP Value')

# %%
# Graph for the expired patients 

plt.figure(figsize=(14, 6))

for patient_id in df_expired['patientunitstayid'].unique():
    patient_data = df_expired[df_expired['patientunitstayid'] == patient_id]
    plt.plot(patient_data['Time'], patient_data['icp'], label=f'Patient {patient_id}')
plt.ylim(5, 40)
plt.title('ICP Values of Expired Patients')
plt.xlabel('Time')
plt.ylabel('ICP Value')


# ----------------- GRAPH LEVELS -------------------

# ---- THE THREE LEVELS OF ALIVE LIST  -------


fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (18,5))

for patient_id in df_alive['patientunitstayid'].unique():
    patient_data = df_alive[df_alive['patientunitstayid'] == patient_id]
    # plt.plot(patient_data['Time'], patient_data['cp'], label=f'Patient {patient_id}')
    if(df_alive.loc[patient_id]['icp'] > 0 and df_alive.loc[patient_id]['icp'] < 25):
        sns.lineplot(ax=axes[0], data = patient_data, x = 'Time', y = 'icp')    
    if(df_alive.loc[patient_id] > 25 and df_alive.loc[patient_id] < 50):
        sns.lineplot(ax=axes[1], data = patient_data, x = 'Time', y = 'icp')
    if(df_alive.loc[patient_id] > 50):
        sns.lineplot(ax=axes[2], data = patient_data, x = 'Time', y = 'icp')

plt.ylim(5, 55)

plt.title('ICP Values of Alive Patients')
plt.xlabel('Time')
plt.ylabel('ICP Value')







# plt.legend()
# def determine_status(patient_id):
#     if patient_id in alive_list:
#         return 'alive'
#     elif patient_id in expired_list:
#         return 'dead'
#     else:
#         return 'unknown'

# df_vitalCopy['status'] = df_vitalCopy['patientunitstayid'].apply(determine_status)
# df_vitalCopy.head()

# %%





# %%
# ----------------very big line break ----------------------






'''
     (\           
    (  \  /(o)\    The code below is the numscaler code, fix later   
    (   \/  ()/ /)  
     (   `;.))'".) 
      `(/////.-'
   =====))=))===() 
     ///'       
    //   PjP/ejm
   '    
'''








# ----------------very big line break ----------------------

# %%
#Normalize Data

scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(dt.drop('Time', axis=1))
normalized_df = pd.DataFrame(normalized_data, columns=dt.drop('Time', axis=1).columns)
normalized_df['Time'] = dt['Time']

plt.figure(figsize=(15, 10))
for col in normalized_df.columns:
    if col != 'Time':
        plt.plot(normalized_df['Time'], normalized_df[col], label=col)

plt.title(f"Patient ID: {patient}", fontsize=16)
plt.xlabel('Time')
plt.ylabel('Normalized Value')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
# %%
fig, axs = plt.subplots(3, 1, figsize=(15, 20), sharex=True)
fig.suptitle(f"Patient ID: {patient}", fontsize=16)

# Plot high-range variables
high_range = ['Unnamed: 0']
for col in high_range:
    axs[0].plot(dt['Time'], dt[col], label=col)
axs[0].legend()
axs[0].set_ylabel('High Range')

# Plot mid-range variables
mid_range = ['systemicsystolic', 'systemicdiastolic', 'systemicmean', 'heartrate']
for col in mid_range:
    axs[1].plot(dt['Time'], dt[col], label=col)
axs[1].legend()
axs[1].set_ylabel('Mid Range')

# Plot low-range variables
low_range = ['temperature', 'sao2', 'respiration', 'icp']
for col in low_range:
    axs[2].plot(dt['Time'], dt[col], label=col)
axs[2].legend()
axs[2].set_ylabel('Low Range')

plt.xlabel('Time')
plt.tight_layout()
plt.show()
# %%
fig, ax1 = plt.subplots(figsize=(15, 10))
fig.suptitle(f"Patient ID: {patient}", fontsize=16)

# Plot high-range variable on secondary y-axis
ax2 = ax1.twinx()
ax2.plot(dt['Time'], dt['Unnamed: 0'], label='Unnamed: 0', color='r')
ax2.set_ylabel('Unnamed: 0', color='r')

# Plot other variables on primary y-axis
for col in dt.columns:
    if col not in ['Time', 'Unnamed: 0']:
        ax1.plot(dt['Time'], dt[col], label=col)

ax1.set_xlabel('Time')
ax1.set_ylabel('Value')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()


# %%


tempNode = dfL_vitals.head
count = 0


# note we also only need the 54 patients from apachescore, not the 4 extra patients. 

while tempNode:  
    dt = tempNode.data
    patient = dt.index.get_level_values('patientunitstayid').unique()[0]
    print(dt.head())

    # Create subplots
    fig, axs = plt.subplots(3, 1, figsize=(15, 20), sharex=True)
    fig.suptitle(f"Patient ID: {patient}", fontsize=16)

    # Plot high-range variables
    high_range = ['Unnamed: 0']
    for col in high_range:
        axs[0].plot(dt['Time'], dt[col], label=col)
    axs[0].legend()
    axs[0].set_ylabel('High Range')

    # Plot mid-range variables
    mid_range = ['systemicsystolic', 'systemicdiastolic', 'systemicmean', 'heartrate']
    for col in mid_range:
        axs[1].plot(dt['Time'], dt[col], label=col)
    axs[1].legend()
    axs[1].set_ylabel('Mid Range')

    # Plot low-range variables
    low_range = ['temperature', 'sao2', 'respiration', 'icp']
    for col in low_range:
        axs[2].plot(dt['Time'], dt[col], label=col)
    axs[2].legend()
    axs[2].set_ylabel('Low Range')

    plt.xlabel('Time')
    plt.tight_layout()
    plt.show()
    
    print(f"Plotted data for patient {patient}")
    
    count += 1
    tempNode = tempNode.next

print(f"Total patients plotted: {count}")



# %%
