# %% intro
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from LinkedListClass import Node, LL
import re
import plotly.graph_objects as go
import streamlit as st

df = pd.read_csv('infsP_results.csv')

pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
# print(df)
# processing df

# Drop columns
df = df.drop(columns=['Unnamed: 0', 'infusionoffset'])

# normalize drugrate


# Sample data

dose_range = df.groupby('drugname')['drugrate'].agg(['min', 'max']).reset_index()
# print(dose_range)                                                            
# multi-index
df = df.set_index(['patientunitstayid', 'drugname'])
# print(df)

ID_List = df.index.get_level_values(0).unique()
# create LL
LL = LL()
for i in range(ID_List.size):
    multiIndex = df[df.index.get_level_values('patientunitstayid') == ID_List[i]]
    print(multiIndex)
    LL.append(multiIndex)

# drugList = df.index.get_level_values(1).unique()
# print(drugList)

# display LL
# LL.display()

# graph each patient's drug graph


node = LL.head
while node:
    df = node.data.reset_index()
    ID = df['patientunitstayid']
    Time =  df['Time']
    drugName = df['drugname']
    drugList = drugName.unique()
    fig = go.Figure()
    for drug in drugList:
        drugRate = df.loc[df['drugname'] == drug, 'drugrate']
        drugTime = df.loc[df['drugname'] == drug, 'Time']

        fig.add_trace(go.Scatter(x=drugTime, y=drugRate, mode='lines+markers', name=f'Patient {ID}: {drug}'))
        fig.update_layout(title='Drug Rate vs Time for each Patient', xaxis_title='Time', yaxis_title='Drug Rate')
        fig.show()
    node = node.next









# %%
import pandas as pd
import matplotlib.pyplot as plt
# Sample data
data = {
    'patient_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
    'drug': ['DrugA', 'DrugA', 'DrugA', 'DrugB', 'DrugB', 'DrugB', 'DrugC', 'DrugC', 'DrugC'],
    'dose': [10, 20, 15, 100, 150, 130, 200, 250, 240],
    'time': pd.date_range(start='2021-01-01', periods=9, freq='D')
}
df = pd.DataFrame(data)

# Calculate the dose range for each drug
dose_range = df.groupby('drug')['dose'].agg(['min', 'max']).reset_index()

# Plot the dose ranges
plt.figure(figsize=(10, 6))
for index, row in dose_range.iterrows():
    plt.plot([row['drug'], row['drug']], [row['min'], row['max']], marker='o', label=f"{row['drug']} range")
plt.xlabel('Drugs')
plt.ylabel('Dose Range')
plt.title('Dose Ranges for Different Drugs')
plt.legend()
plt.show()
# %%
