# %%
import pandas as pd
import numpy as np 

# %%

wce_merged_model = pd.read_csv('wce_merged_model.csv')
wce_merged_model.head()

# %%


# delete all rows without any values for auc #change
