# %%
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("labsP.csv")


print("The maximum is ", df['labresultoffset'].max())


df.head(100)

#labresultoffset : minutes for the patient at the icu. 
# if negative, its from before they were at the icu. we can get rid of those 
# we can divide by 60 for hours only 

# for each hour: 
# look at the peak value for that feature or hour. if a hundred recordings for the hour, have the peak, lowest, and the median 
# then only one time parameter which is hour, and the 3 values for each of the features. 
# first hour offset is 0 to 60. turn that into an integer and make it 1. then all the values between 0 to 60 will be 
# described using 3 parameters. 



# Necessary lab params: 

# 

# %%
df.loc[df["labresultoffset"] < 0, 'labresultoffset'] = np.NaN
# data post cvp clean
df.head()

# %%
sns.heatmap(pd.isnull(df), cmap = 'viridis')

# %%
for col in df.columns: 
    df = df.dropna(subset=[col])

# note if this column is important. 
df = df.drop(columns=['labmeasurenameinterface', 'Unnamed: 0'])


# df.drop_duplicates()
df = df.sort_values('labresultoffset')
df = df.sort_values('patientunitstayid')

# df.drop_duplicates()

df.head()

# %%
# df.loc['labresultoffset', :] = df.loc['labresultoffset', :] / 60

df["labresultoffset"] = df["labresultoffset"].div(60)
df['labresultoffset'] = df['labresultoffset'].astype(int)

print("The maximum is ", df['labresultoffset'].max())


df.head(1000000000000)

# %%
for col in df.columns: 
    df = df.dropna(subset=[col])

df = df.sort_values('labresultoffset')

# df.drop_duplicates()

df.describe()

# %%
df.sort_values(by=['patientunitstayid','labname','labresultoffset', 'labresult'])

# %%
means = df['labresult'].groupby([df['patientunitstayid'], df['labresultoffset'], df['labname']]).mean()

mins = df['labresult'].groupby([df['patientunitstayid'], df['labresultoffset'], df['labname']]).min()

maxes = df['labresult'].groupby([df['patientunitstayid'], df['labresultoffset'], df['labname']]).max()

counts = df['labresult'].groupby([df['patientunitstayid'], df['labresultoffset'], df['labname']]).count()



# %%
means.head()

# %%
maxes.head()

# %%
mins.head()

# %%
counts.head()

# %%
#all scatter plots

# cols = ['patientunitstayid', 'means', 'mins', 'maxes']
# fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))
# axes = axes.flatten()

# for i, col in enumerate(cols):
#     sns.scatterplot(data=df, x='patientunitstayid', y=cols, ax=axes[i])
#     axes[i].set_title(col)

# plt.tight_layout()
# plt.show()

# for (k1, k2), group in df.groupby(["key1", "key2"]):
 #  ....:     print((k1, k2))
#   ....:     print(group)

# %%


# %%


# %%



