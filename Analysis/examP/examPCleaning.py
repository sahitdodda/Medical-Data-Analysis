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

grouped = { 
    'notes/Progress Notes/Physical Exam/Physical Exam/Constitutional/Vital Sign and Physiological Data/FiO2%/FiO2%' : 'FiO2%', 
    'notes/Progress Notes/Physical Exam/Physical Exam/Constitutional/Vital Sign and Physiological Data/O2 Sat%/O2 Sat% Current' : 'O2 Sat%', 
    'notes/Progress Notes/Physical Exam/Physical Exam/Constitutional/Vital Sign and Physiological Data/PEEP/PEEP' : 'PEEP',

    # group 1, spontaneous and ventilated
    'notes/Progress Notes/Physical Exam/Physical Exam/Constitutional/Vital Sign and Physiological Data/Resp Mode/' : 'respMode',

    'notes/Progress Notes/Physical Exam/Physical Exam/Constitutional/Weight and I&O/Weight (kg)/Admission' : 'kgAdmission',

    # group 2 
    'notes/Progress Notes/Physical Exam/Physical Exam/Head and Neck/Eyes/Pupils/(reaction)/' : 'pupilReact',

    # group 3
    'notes/Progress Notes/Physical Exam/Physical Exam/Head and Neck/Eyes/Pupils/(symmetry)/' : 'pupilSymmetry',

    # group 4
    'notes/Progress Notes/Physical Exam/Physical Exam/Neurologic/Cranial Nerves/' : 'crainNerves',

    # group 5 
    'notes/Progress Notes/Physical Exam/Physical Exam/Neurologic/GCS/' : 'neuroGCS',

    # group 6 
    'notes/Progress Notes/Physical Exam/Physical Exam/Neurologic/GCS/Eyes Score' : 'neuroGCS_Eyes',

    # group 7 
    'notes/Progress Notes/Physical Exam/Physical Exam/Neurologic/GCS/Mental Status/Affect/' : 'mentalAffect',

    # group 8 
    'notes/Progress Notes/Physical Exam/Physical Exam/Neurologic/GCS/Mental Status/Level of Consciousness/' : 'levelConsciousness', 

    # group 9 
    'notes/Progress Notes/Physical Exam/Physical Exam/Neurologic/GCS/Mental Status/Orientation/unable to assess orientation' : 'unableOrient',

    # group 10 
    'notes/Progress Notes/Physical Exam/Physical Exam/Neurologic/GCS/Motor Score/' : 'neuroGCS_Motor',

    # group 11
    'notes/Progress Notes/Physical Exam/Physical Exam/Neurologic/GCS/Score/' : 'neuroGCS_Score',

    # group 12
    'notes/Progress Notes/Physical Exam/Physical Exam/Neurologic/GCS/Verbal Score/' : 'neuroGCS_Verbal_Score',

    'notes/Progress Notes/Physical Exam/Physical Exam/Neurologic/Motor/decreased strength' : 'motorDecreased',

    'notes/Progress Notes/Physical Exam/Physical Exam/Neurologic/Reflexes/Abnormal Reflex/diffusely' : 'reflexDiffused',

    'notes/Progress Notes/Physical Exam/Physical Exam/Neurologic/Reflexes/decreased ' : 'reflexDecreased',

    'notes/Progress Notes/Physical Exam/Physical Exam/Neurologic/Sensation/no response to pain ' : 'noPain', 

    # group 13 
    'notes/Progress Notes/Physical Exam/Physical Exam/Pulmonary/Airway/' : 'pulmonaryAirway'

}


def replace_item(item):
    for pattern, actual in grouped.items(): 
        if re.search(pattern, item):
            return actual 
    return item 


df['physicalexampath'] = df['physicalexampath'].apply(replace_item)

# %%

# for now setting to nans and dropping all nans 
df.loc[df['physicalexampath'].contains('notes')] = np.NaN


df.head(100000000000000000000)