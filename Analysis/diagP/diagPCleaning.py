# %%
import pandas as pd
 
# make relative paths later
df = pd.read_csv('diagP.csv')


# look at jupyter variable files to actually see the whole thing
df.head() 


# %%
df = df.drop(columns=['icd9code'])
df.head()


# %%
# delete rows with activeupon discharge = false 

df = df.loc[df['activeupondischarge'] != False]


# %%
df = df.loc[df['diagnosispriority'].isin(['Primary', 'Major'])]

df.head()



