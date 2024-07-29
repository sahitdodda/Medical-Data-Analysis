# %%
import pandas as pd
import numpy as np 

# %%
wce_merged_edited = pd.read_csv('wce_merged_edited.csv')
wce_merged_edited = wce_merged_edited.drop(columns=['AgeGr', 'Female', 'actualiculos', 'AdmGCS', 'MortDay7'])
wce_merged_model = pd.read_csv('wce_merged_model.csv')

# %%

pList = wce_merged_model['patientunitstayid_x'].unique()
# find the patient who has one for the value of compoutcome
pList = wce_merged_model['patientunitstayid_x'].unique()
newList = []
# set all of the patient compoutcome to 0, set last row to 1
for p in pList:
    rows = wce_merged_model[wce_merged_model['patientunitstayid_x'] == p]
    if rows['CompOutcome'].iloc[0] == 1:
        rows['CompOutcome'] = 0 # set all to 0
        rows['CompOutcome'].iloc[len(rows)-1] = 1 # manually set last row to 1
    newList.append(rows)

wce_merged_model_NEW = pd.concat(newList) # adjusted row values


# --------------------------------------------------------------------------------------------

pList = wce_merged_edited['patientunitstayid'].unique()
newList = []
# iterate through each patient
for p in pList:
    rows = wce_merged_edited[wce_merged_edited['patientunitstayid'] == p]
    i = len(rows) - 1
    end = 0
    # loop backwards until we find the 'end' row
    while rows['Total.AUC.for.Hour'].iloc[i] == 0:
        i-=1
    # save only the rows until 'end row'
    newList.append(rows.iloc[:i+1])

# save new dataframe
wce_merged_edited_NEW = pd.concat(newList) # deleted rows
wce_merged_edited_NEW.head()

# %%
wce_merged_model_NEW.to_csv('wce_merged_model_NEW.csv')
wce_merged_edited_NEW.to_csv('wce_merged_edited_NEW.csv')

wce_merged_model_NEW.rename(columns={'patientunitstayid_x' : 'patientunitstayid'}, inplace=True)
wce_merged_FINAL = pd.merge(wce_merged_edited_NEW, wce_merged_model_NEW, on=['patientunitstayid', 'Hour'])

# common_columns = set(wce_merged_edited_NEW.columns).intersection(set(wce_merged_model_NEW.columns))
# print(common_columns)

print(wce_merged_model_NEW.shape)
print(wce_merged_edited_NEW.shape)
print(wce_merged_FINAL.shape)
# %%

wce_merged_FINAL.to_csv('wce_merged_FINAL.csv')