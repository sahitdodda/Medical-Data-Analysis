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
from IPython.display import display
from datetime import timedelta
import plotly.graph_objects as go
import streamlit as st

# %%

final_AUC_DF = pd.read_csv('finalAUC.csv')
icp_day1_DF = pd.read_csv('icp_first_24.csv')

final_merged = pd.merge(final_AUC_DF, icp_day1_DF, on='patientunitstayid')

final_merged.head(100000000)


# %%

df_analysis = final_merged[['patientunitstayid', 'AgeGr', 'ISS', 'predictedicumortality', 'actualicumortality', 'AdmGCS', 'CompOutcome', 'AUC24cum', 'AUC25cum', 'AUC26cum', 'AUC27cum', 'AUC28cum', 'AUC29cum', 'AUC30cum', 'AUC31cum', 'AUC32cum', 'AUC33cum', 'AUC34cum', 'AUC35cum', 'AUC36cum', 'AUC37cum', 'AUC38cum', 'AUC39cum', 'AUC40cum', 'AUC41cum', 'AUC42cum', 'AUC43cum', 'AUC44cum', 'AUC45cum', 'AUC46cum', 'AUC47cum', 'AUC48cum', 'AUC49cum', 'AUC50cum', 'AUC51cum', 'AUC52cum', 'AUC53cum', 'AUC54cum', 'AUC55cum', 'AUC56cum', 'AUC57cum', 'AUC58cum', 'AUC59cum', 'AUC60cum', 'AUC61cum', 'AUC62cum', 'AUC63cum', 'AUC64cum', 'AUC65cum', 'AUC66cum', 'AUC67cum', 'AUC68cum', 'AUC69cum', 'AUC70cum', 'AUC71cum', 'AUC72cum']]

df_analysis.head()

# %%

