# %%

import pandas as pd 
import numpy as np

df = pd.read_csv('bloodpressures.csv')

df.head()
# %%

# calculate mean value for each patient every hour 
# Group by patientunitstayid and Hour, then calculate remaining features 
means = df.groupby(['patientunitstayid', 'Hour'])[['cpp', 'systemicsystolic']].mean().reset_index()

# %%

# review the head 
means.head()

# %%

# merge by the same groupby columns, and add a _mean suffix. 
merged_df = pd.merge(df, means, on=['patientunitstayid', 'Hour'], suffixes=('', '_mean'))

merged_df.head()

# %%

merged_df.to_csv('bloodpressures_means.csv')