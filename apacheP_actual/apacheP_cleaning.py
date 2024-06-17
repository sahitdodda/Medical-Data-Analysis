# %%
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt

df = pd.read_csv('apacheP.csv')
df.head(1000)

# %%
sns.heatmap(pd.isnull(df), cmap = 'viridis')

# %%
df.isna().sum()

# %%
df.isna().sum().sum()

# %%
df = df.set_index(['patientunitstayid'])

df.head(100000000)


# %%
df.shape

# %%
df.loc[193629]

# %%
# get list of the unique values using get_level_values of patientunitstayid

unique_patient_ids = df.index.get_level_values('patientunitstayid').unique()

print(unique_patient_ids)
print(len(unique_patient_ids))

# %%
# iterates through list of unique patient ID's, prints individual stats for each patientID

for patient_id in unique_patient_ids:
    print(f"Processing patient ID: {patient_id}")
    patient_data = df.loc[patient_id]
    print(patient_data)


