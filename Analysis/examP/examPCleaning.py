# %%
import pandas as pd
import numpy as np 
import re 

df = pd.read_csv('examP.csv')

df.head()


# %%
# techhnically there are no negatives 
df = df.loc[df['physicalexamoffset'] > 0]
df['Time'] = df['physicalexamoffset'].div(60)

df.head()

# %%



