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

df_vitalsP = df_vitalsP.drop(columns=['Unnamed: 0', 'observationoffset', 'Day', 'Hour'])
# patient 1082792 literally has only one data point :) 

# df_vitalsP = df_vitalsP.loc[df_vitalsP['patientunitstayid'].isin(['30989', '11302920'])]




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
df_vitalsP = df_vitalsP.set_index(['patientunitstayid', 'Time'])


# %%

# check that it worked 
# df_vitalsP.loc[193629]

# %%

unique_patient_ids = df_vitalsP.index.get_level_values('patientunitstayid').unique()
print(unique_patient_ids)
print(len(unique_patient_ids))


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
    
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    
    count += 1
    tempNode = tempNode.next

# %%

# This code is in fact better, the num d type thing is a bit of a problem and doesn't show data 
# for patient 9 
tempNode = dfL_vitals.head
count = 0

while tempNode and count < 12:  # Limit to 5 patients for example
    dt = tempNode.data
    patient = dt.index.get_level_values('patientunitstayid').unique()[0]
    
    columns = [col for col in dt.columns]
    
    plt.figure(figsize=(15, 10))
    plt.title(f"Patient ID: {patient}", fontsize=16)
    
    print(f"Plotting data for patient {patient}")
    
    for column in columns:
        sns.scatterplot(data=dt, x='Time', y=column, label=column)
    
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    count += 1
    tempNode = tempNode.next

print(f"Total patients plotted: {count}")






# %%
# ----------------very big line break ----------------------






'''
     (\           
    (  \  /(o)\    The code below is the numscaler code  
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
