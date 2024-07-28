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




# NOTE: 
    # WCE_MERGED was generating using the commented out methodology
    # However, it was edited to no longer contain negatives, so it is read in directly.
    # The methodology is still here in case things need to be edited later on  


# %%

# final_AUC_DF = pd.read_csv('finalAUC.csv')
# icp_day1_DF = pd.read_csv('icp_first_24.csv')


# final_merged = pd.merge(final_AUC_DF, icp_day1_DF, on='patientunitstayid')

# final_merged.head(100000000)

# %%

# df_analysis = final_merged[['patientunitstayid', 'AgeGr', 'ISS', 'predictedicumortality', 'actualicumortality', 'AdmGCS', 'CompOutcome', 'AUC24cum', 'AUC25cum', 'AUC26cum', 'AUC27cum', 'AUC28cum', 'AUC29cum', 'AUC30cum', 'AUC31cum', 'AUC32cum', 'AUC33cum', 'AUC34cum', 'AUC35cum', 'AUC36cum', 'AUC37cum', 'AUC38cum', 'AUC39cum', 'AUC40cum', 'AUC41cum', 'AUC42cum', 'AUC43cum', 'AUC44cum', 'AUC45cum', 'AUC46cum', 'AUC47cum', 'AUC48cum', 'AUC49cum', 'AUC50cum', 'AUC51cum', 'AUC52cum', 'AUC53cum', 'AUC54cum', 'AUC55cum', 'AUC56cum', 'AUC57cum', 'AUC58cum', 'AUC59cum', 'AUC60cum', 'AUC61cum', 'AUC62cum', 'AUC63cum', 'AUC64cum', 'AUC65cum', 'AUC66cum', 'AUC67cum', 'AUC68cum', 'AUC69cum', 'AUC70cum', 'AUC71cum', 'AUC72cum']]

# df_analysis.head()

# %%
# Expand 52 rows by 49 each 

# auc_results_df = pd.read_csv('AUC_RESULTS_Hourly.csv')
# row_ids = [i for i in range(2548)]
# auc_results_df['row_id'] = row_ids

# ptFeatures_temp = pd.read_csv('ptFeatures.csv')
# ptFeatures_df = ptFeatures_temp.loc[ptFeatures_temp.index.repeat(49)].reset_index(drop=True)

# ptFeatures_df['row_id'] = row_ids

# print(len(auc_results_df))
# print(len(ptFeatures_df))


# %%
# from hourly table, pick up total AUC for hour 

# wce_merged = pd.merge(auc_results_df, ptFeatures_df, on='row_id')
# wce_merged.head(100000)

# wce_merged.to_csv('wce_merged.csv')


wce_merged = pd.read_csv('wce_merged.csv')

wce_merged['End Hour'] = wce_merged['Hour'] + 1

columns_filter = ['patientunitstayid_x', 'Hour', 'End Hour', 'predModel', 'actualicumortality', 'actualiculos', 'AdmGCS', 'CompOutcome', 'Total.AUC.for.Hour', 'num_spike_20']


wce_merged_model = wce_merged[columns_filter]

wce_merged_model.head(100000)

wce_merged_model.to_csv('wce_merged_model.csv')


 