# %% intro
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import missingno as msno  # Import missingno library
from LinkedListClass import Node, LL
import re
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('infsP_results.csv')
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns

# Drop columns
df = df.drop(columns=['Unnamed: 0', 'infusionoffset'])
# convert object type to float
print("Data types of each column:\n", df.dtypes)
df['drugrate'] = pd.to_numeric(df['drugrate'], errors='coerce')
print("Data types of each column:\n", df.dtypes)
rowList = df[df['drugrate'].isnull()].index.tolist()
# nan graph and nan rows
msno.matrix(df)
plt.show()
print(len(rowList))
print(rowList)
# drop rows with nan values
df = df.dropna()
# normalize drugrate for each drug
scaler = MinMaxScaler()

drugList = df['drugname'].unique()
for drug in drugList:

    drug_df = df[df['drugname'] == drug]
    drug_df['drugrate'] = scaler.fit_transform(drug_df[['drugrate']])
    df.loc[df['drugname'] == drug, 'drugrate'] = drug_df['drugrate']
    print(df[df['drugname'] == drug])