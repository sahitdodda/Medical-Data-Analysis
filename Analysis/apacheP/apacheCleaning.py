# %%
import pandas as pd 
import numpy as np 
df = pd.read_csv('apacheP.csv')

df.head()

# %%
df.loc[df["apachescore"] < 1, 'apachescore'] = np.NaN


