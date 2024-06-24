# %%
# just combining the script with what we had already for labsP 

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

df = pd.read_csv('labsP.csv')

df.head(1000000)

# %%

grouped = ['albumin', 'ALT (SGPT)', 'AST (SGOT)', 
           'bedside glucose', 'BUN', 'creatinine', 
           'direct bilirubin', 'fibrinogen', 'FiO2', 
           'glucose', 'Hct', 'Hgb', 'ionized calcium', 
           'lactate', 'O2 sat (%)', 'paCO2', 'paO2', 'Peak Airway/Pressure', 'PEEP', 'pH', 'platelets x 1000', 'PT – INR', 
           'Respiratory Rate', 'sodium', 'Spontaneous Rate', 'Vent Rate']


df = df[df['labname'].isin(grouped)]
df.head(100000000)

# %%
# df.drop_duplicates()
df = df.sort_values('labresultoffset')
df = df.sort_values('patientunitstayid')


# %%

sns.heatmap(pd.isnull(df), cmap = 'viridis')

# %%
df.isna().sum()

# %%
df.isna().sum().sum()

# %%
df.loc[df["labresultoffset"] < 0, 'labresultoffset'] = np.NaN

# drop nans for now, impute later pretty please
for col in df.columns: 
    df = df.dropna(subset=[col])

# %%
# set to be in terms of hours 

df["labresultoffset"] = df["labresultoffset"].div(60)
df['labresultoffset'] = df['labresultoffset'].astype(int)

print("The maximum is ", df['labresultoffset'].max())

df.head(1000000000000)

# %%
df.sort_values('labresultoffset')


means = df['labresult'].groupby([df['patientunitstayid'], df['labresultoffset'], df['labname']]).mean()

mins = df['labresult'].groupby([df['patientunitstayid'], df['labresultoffset'], df['labname']]).min()

maxes = df['labresult'].groupby([df['patientunitstayid'], df['labresultoffset'], df['labname']]).max()

counts = df['labresult'].groupby([df['patientunitstayid'], df['labresultoffset'], df['labname']]).count()

df.sort_values('labresultoffset')


# %%
means.head()

# %%

meansArray = means.values
print(meansArray)
means.head()

# %%
maxesArray = maxes.values
print(maxesArray)
maxes.head()

# %%
minsArray = mins.values
print(minsArray)
mins.head()

# %%
countsArray = counts.values
print(countsArray)
counts.head()

# %%
df.sort_values('labresultoffset')
df = df.set_index(['patientunitstayid'])

df.head()


# %%
df.shape

# %%
df.loc[193629]

# %%

unique_patient_ids = df.index.get_level_values('patientunitstayid').unique()

print(unique_patient_ids)
print(len(unique_patient_ids))

# %%
for patient_id in unique_patient_ids:
    print(f"Processing patient ID: {patient_id}")
    patient_data = df.loc[patient_id]
    print(patient_data)

# %%
# now putting the actual calculations into the new dataframe

class Node: 
    def __init__(self, data=None):
        self.data = data 
        self.next = None

class LL: 
    def __init__(self):
        self.head = None # head of list is none
    def append(self, data):
        new_node = Node(data) # construct node from data 
        if self.head is None: 
            self.head = new_node # if list empty, new node is head 
        else:
            current = self.head
            while current.next: # loop to get to last element of the linked list
                current = current.next  
            current.next = new_node
    def display(self):
        current = self.head
        while current: 
            print(current.data)
            current = current.next # displaying the list, not equiv to concatenation
    def length(self):
        current = self.head
        count = 0
        while current: 
            count += 1
            current = current.next # displaying the list, not equiv to concatenation
        return count
    def find(self, patientunitstayid):
        current = self.head
        while current:
            if current.data.index.values[0] == patientunitstayid:
                return current.data
            current = current.next
        return None

# %%
dfList =  LL()

df1 = df.loc[193629]

for patient_id in unique_patient_ids:
    dfIter = df.loc[patient_id]
    dfList.append(dfIter)

dfList.display()
print(dfList.length())



# %%
tempNode = dfList.head
count = 0
while tempNode:
    # do stuff
    #tempNode.data.sort_values('labresultoffset')
    dt = tempNode.data
    means = dt['labresult'].groupby([dt['labresultoffset'], dt['labname']]).mean()
    # print(f"patient {count}")
    
    means = means.reset_index()

    labresultoffset = means['labresultoffset']
    values = means['labresult']

    plt.figure()
    sns.kdeplot(data = dt, x = labresultoffset, y = values, shade=True)
    
    # print(tempNode.data.head(1000000000000000))
    count += 1
    tempNode = tempNode.next



