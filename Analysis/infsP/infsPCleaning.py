# %%
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

df = pd.read_csv('infsP.csv')

df.head()


# %%

df = df.loc[df['infusionoffset'] > 0]
df['Time'] = df['infusionoffset'].div(60)

df.head()


# do we want to create notes for each group? 
# be sure to ask about how we should do this.
