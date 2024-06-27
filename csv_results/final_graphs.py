# working on the final levels of visualization

# %%
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.linear_model import LinearRegression
import missingno as mno

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
df_vitalsP = df_vitalsP.fillna(df_vitalsP.median())
df_apache = df_apache.dropna()
df_infs = df_infs.dropna()

# %%
mno.matrix(df_apache, figsize=(20, 6))

# %%

# Now we make the giant linked list structure for um everything
# df_vitalsP = df_vitalsP.sort_values(by=['patientunitstayid', 'Time'])


df_vitalsP = df_vitalsP.sort_values(by=['patientunitstayid', 'Time'])
df_vitalsP = df_vitalsP.set_index(['patientunitstayid'])


# %%

# check that it worked 
df_vitalsP.loc[193629]

# %%

unique_patient_ids = df_vitalsP.index.get_level_values('patientunitstayid').unique()
print(unique_patient_ids)
print(len(unique_patient_ids))


# %% 

dfL_vitals = LL()

for patient_id in unique_patient_ids: 
    dfIter = df_vitalsP.loc[patient_id]
    dfL_vitals.append(dfIter)

dfL_vitals.display()
print(dfL_vitals.length())


 # %%

tempNode = dfL_vitals.head
count = 0
while tempNode: 
    dt = tempNode.data 
    #shows which one we're working with 

    patient = dt.index.get_level_values('patientunitstayid').unique()
    columns = dt.loc[patient].columns
    plt.figure()
    # plt.ylim(top = 180)
    # plt.ylim(bottom = 110)
    print(patient)
    for column in columns: 
        sns.scatterplot(data = dt, x = time_list, y = icp_list, label='icp')
        # sns.scatterplot(data = dt, x = time_list, y = column)

    count += 1
    tempNode = tempNode.next
        



# %%
