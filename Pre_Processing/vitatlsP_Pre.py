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

import plotly.graph_objects as go
import streamlit as st

import sys 
sys.path.append('LinkedListClass.py')
from LinkedListClass import Node, LL

# %% (Pre-processing) dropping columns from df_vitalsP
df_vitalsP = pd.read_csv(r'csv_results\vitalsP.csv')

df_vitalsP = df_vitalsP.drop(columns=['Unnamed: 0', 'observationoffset', 'Day', 'Hour', 'systemicdiastolic', 'systemicsystolic'])
#%% (FIX)(Pre-processing) imputation for vitalsP
mno.matrix(df_vitalsP, figsize=(20, 6)) # displays NaN's in df_vitalsP
df_vitalsP = df_vitalsP.fillna(method='ffill') #forward fill (not ideal)

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