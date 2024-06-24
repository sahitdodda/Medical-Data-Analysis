# %%
import pandas as pd
import numpy as np 
import re 

import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('examP.csv')

df.head()


# %%

sns.heatmap(pd.isnull(df), cmap = 'viridis')



# %%
# techhnically there are no negatives 
df = df.loc[df['physicalexamoffset'] > 0]
df['Time'] = df['physicalexamoffset'].div(60)

df.head()

# %%

grouped = { 
    r'notes/Progress Notes/Physical Exam/Physical Exam/Constitutional/Vital Sign and Physiological Data/FiO2%/FiO2%' : 'FiO2%', 
    r'notes/Progress Notes/Physical Exam/Physical Exam/Constitutional/Vital Sign and Physiological Data/O2 Sat%/O2 Sat% Current' : 'O2 Sat%', 
    r'notes/Progress Notes/Physical Exam/Physical Exam/Constitutional/Vital Sign and Physiological Data/PEEP/PEEP' : 'PEEP',

    # group 1, spontaneous and ventilated
    r'notes/Progress Notes/Physical Exam/Physical Exam/Constitutional/Vital Sign and Physiological Data/Resp Mode/' : 'respMode',

    # doesn't exist!!!!!!
    r'notes/Progress Notes/Physical Exam/Physical Exam/Constitutional/Weight and I&O/Weight (kg)/Admission' : 'kgAdmission',

    # group 2 -> doesn't exist!!!!!!
    r'notes/Progress Notes/Physical Exam/Physical Exam/Head and Neck/Eyes/Pupils/(reaction)/' : 'pupilReact',

    # group 3 -> doesn't exist!!!!!
    r'notes/Progress Notes/Physical Exam/Physical Exam/Head and Neck/Eyes/Pupils/(symmetry)/' : 'pupilSymmetry',

    # group 4
    r'notes/Progress Notes/Physical Exam/Physical Exam/Neurologic/Cranial Nerves/' : 'crainNerves',

    # group 5 
    r'notes/Progress Notes/Physical Exam/Physical Exam/Neurologic/GCS/' : 'neuroGCS',

    # group 6 -> !!!!
    r'notes/Progress Notes/Physical Exam/Physical Exam/Neurologic/GCS/Eyes Score/' : 'neuroGCS_Eyes',

    # group 7 !!!!
    r'notes/Progress Notes/Physical Exam/Physical Exam/Neurologic/GCS/Mental Status/Affect/' : 'mentalAffect',

    # group 8 
    r'notes/Progress Notes/Physical Exam/Physical Exam/Neurologic/GCS/Mental Status/Level of Consciousness/' : 'levelConsciousness', 

    # group 9 -> doesn't exist!!!!!
    r'notes/Progress Notes/Physical Exam/Physical Exam/Neurologic/GCS/Mental Status/Orientation/unable to assess orientation/' : 'unableOrient',

    # group 10 !!!!
    r'notes/Progress Notes/Physical Exam/Physical Exam/Neurologic/GCS/Motor Score/' : 'neuroGCS_Motor',

    # group 11!!!!!
    r'notes/Progress Notes/Physical Exam/Physical Exam/Neurologic/GCS/Score/' : 'neuroGCS_Score',

    # group 12!!!!
    r'notes/Progress Notes/Physical Exam/Physical Exam/Neurologic/GCS/Verbal Score/' : 'neuroGCS_Verbal_Score',

    # does not exist !!!!
    r'notes/Progress Notes/Physical Exam/Physical Exam/Neurologic/Motor/decreased strength/' : 'motorDecreased',
    # not exist !!!!
    r'notes/Progress Notes/Physical Exam/Physical Exam/Neurologic/Reflexes/Abnormal Reflex/diffusely/' : 'reflexDiffused',
    # not exist !!!!
    r'notes/Progress Notes/Physical Exam/Physical Exam/Neurologic/Reflexes/decreased/ ' : 'reflexDecreased',
    #!!!!
    r'notes/Progress Notes/Physical Exam/Physical Exam/Neurologic/Sensation/no response to pain/ ' : 'noPain', 

    # group 13 
    r'notes/Progress Notes/Physical Exam/Physical Exam/Pulmonary/Airway/' : 'pulmonaryAirway'

}


def replace_item(item):
    for pattern, actual in grouped.items(): 
        if re.search(pattern, item):
            return actual 
    return item 




df['physicalexampath'] = df['physicalexampath'].apply(replace_item)

# %%

# for now setting to nans and dropping all nans 


df.loc[df['physicalexampath'].str.contains('notes', na=False), 'physicalexampath'] = np.NaN

df.head(100000000000000000000)


# %%


df.dropna(subset=['physicalexampath'], inplace=True)
pathList = df['physicalexampath'].tolist()
print(pathList)

# %%

# Now, we attempt to do some level of multindexing 

'''
 PID -> TIME -> 1  FIO2
                2  FIO2
                3  FIO2
                _______
                1  O2sat
                2  ... 
                3                
'''

# the correct method should approx be sortby, which should create a multi index 
# kind of exactly in the fashion shown above. 


# temporary solution 

# tells you where stuff moved in context of the original
# df.reset_index(inplace=True)

df = df.sort_values(by=['patientunitstayid', 'Time', 'physicalexampath'])
# df.set_index(['patientunitstayid', 'Time', 'physicalexampath'], inplace=True)
df.head(100000000)



# %%

# example plots to make sure nothing be broken 

plt.figure()
# cool test woooo funny graph
sns.barplot(data=df, x='Time', y='physicalexampath')






# %%

catList = df['physicalexampath'].tolist()
print(catList)



# %% 
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(catList):
    sns.scatterplot(data=df, x='Time', y=catList, ax=axes[i])
    axes[i].set_title(catList)

plt.tight_layout()
plt.show()



# look into LL idea for plotting each 
# patientid!! and then also each lab. 


# %%

