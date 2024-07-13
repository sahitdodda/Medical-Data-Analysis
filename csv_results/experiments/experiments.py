# %% 
# %%
import pandas as pd
import numpy as np 
import missingno 

# plotting and regular expressions (if we need for some reason)
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import re

# machine learning libraries -- using randomforest for now 
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import cv
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# If I want to use grid or random
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import scipy.stats as stats


# %%

df_vitals = pd.read_csv('EXPERIMENTS_AAHH.csv')
df_auc = pd.read_csv('EXPERIMENTS_AUC_RANGES.csv')
df_days = pd.read_csv('EXPERIMENTS_RANGE.csv')

df_vitals = df_vitals.drop(columns=['temperature'])
df_vitals_agg = df_vitals.groupby('patientunitstayid').mean().reset_index()
df_merged = pd.merge(df_vitals_agg, df_days, on='patientunitstayid')

# %%

df_merged.head()

# %%

# treat each day as a feature and the rows or icp_loads as 'labels'

features = df_merged.drop(columns=['sao2', 'heartrate', 'respiration'])

X_data = features 
y_data = df_merged['Day 1']

X_train_data, X_test_data, Y_train_data, Y_test_data = train_test_split(X_data, y_data, test_size=0.2, random_state=42)


# %%

dtrain_data = xgb.DMatrix(X_train_data, label=Y_train_data)
dtest_data = xgb.DMatrix(X_test_data, label=Y_test_data)

# %%

param_dist = {
    'max_depth': stats.randint(3, 10),
    'learning_rate': stats.uniform(0.01, 0.1),
    'subsample': stats.uniform(0.5, 0.5),
    'n_estimators':stats.randint(50, 200)
}

# Create the XGBoost model object
xgb_model = xgb.XGBClassifier()

# Create the RandomizedSearchCV object
random_search = RandomizedSearchCV(xgb_model, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy')

# Fit the RandomizedSearchCV object to the training data
random_search.fit(X_train_data, Y_train_data)

best_params_data = random_search.best_params_


# Print the best set of hyperparameters and the corresponding score
print("Best set of hyperparameters for data: ", random_search.best_params_)
print("Best score for data: ", random_search.best_score_)
