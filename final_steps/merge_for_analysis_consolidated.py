# NOTE: This file is a combination of merge_for_analysis.py and fixes.py 

# %%
import pandas as pd
import numpy as np

# %%

# NOTE: 
    # There will be some weirdness with notation, but here is an explanation: 
    # wce_merged will be created from final_auc_df

    # Features to be kept in accordance with everything 
        # patientunitstayid, (AUC_RESULTS_hourly_168)
        # Hour, End Hour (manip AUC_RESULTS_hourly_168)
        # Total AUC for Hour (AUC_RESULTS_hourly_168)

# %%

auc_results_df = pd.read_csv('AUC_RESULTS_hourly_168.csv')
row_ids = [i for i in range(7540)]
auc_results_df['row_ids'] = row_ids

ptFeatures_temp = pd.read_csv('ptFeatures.csv')

ptFeatures_df = ptFeatures_temp.loc[ptFeatures_temp.index.repeat(145)].reset_index(drop=True)
ptFeatures_df['row_ids'] = row_ids

ptFeatures_df.head()

# %%

# Ensure that both are of equal length 
print(len(auc_results_df))
print(len(ptFeatures_df))

# %%

wce_merged = pd.merge(auc_results_df, ptFeatures_df, on=['patientunitstayid', 'row_ids'])
wce_merged['End Hour'] = wce_merged['Hour'] + 1
wce_merged.head(10000000)

# %%

columns_filter = ['patientunitstayid', 'Hour', 'End Hour', 'predModel', 'actualicumortality', 'actualiculos', 'AdmGCS', 'CompOutcome', 'Total AUC for Hour', 'num_spike_20', "AgeGr", "Female"]

# main thing we will be working with 
wce_merged_model = wce_merged[columns_filter]
# print(wce_merged_model)


pList = wce_merged_model['patientunitstayid'].unique()
newList = []

# set all of the patient compoutcome to 0, set last row to 1
for p in pList:
    patient = wce_merged_model[wce_merged_model['patientunitstayid'] == p]
    if patient.iloc[0]['CompOutcome'] == 1: # iloc for selection, loc for assignment
        patient.loc[:, 'CompOutcome'] = 0
        patient.loc[patient.index[-1], 'CompOutcome'] = 1
    newList.append(patient)
wce_merged_model_NEW = pd.concat(newList) # adjusted row values

# --------------------------------------------------------------------------------------------

pList = wce_merged_model_NEW['patientunitstayid'].unique()
newList = []


# iterate through each patient
for p in pList:
    patient = wce_merged_model[wce_merged_model['patientunitstayid'] == p]
    print(patient)
    i = len(patient) - 1
    # loop backwards until we find the 'end' row
    while i >= 0 and patient.iloc[i]['Total AUC for Hour'] == 0:
        i -= 1

    print(i)
    # save only the rows until 'end row'
    newList.append(patient.iloc[:i+1])
    print(patient)
    break

# comp_outcome_counts = wce_merged_model_NEW['CompOutcome'].value_counts()
# print("Number of 0's:", comp_outcome_counts.get(0, 0))
# print("Number of 1's:", comp_outcome_counts.get(1, 0))


# %%


# # save new dataframe
# wce_merged_edited_NEW = pd.concat(newList) # deleted rows

# print(wce_merged_edited_NEW)

# %%

# grab predictedoutcome from wce_merged_FINAL.csv
wce_pred_outcome = pd.read_csv('wce_merged_FINAL.csv')
wce_pred_outcome = wce_pred_outcome[['patientunitstayid', 'predictedoutcome']]
wce_pred_outcome = wce_pred_outcome.drop_duplicates(subset='patientunitstayid')
wce_pred_outcome = wce_pred_outcome.loc[wce_pred_outcome.index.repeat(145)].reset_index(drop=True)

print(len(wce_pred_outcome))
print(wce_merged_edited_NEW)