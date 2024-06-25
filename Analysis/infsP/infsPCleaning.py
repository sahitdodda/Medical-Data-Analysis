# %%
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import re

df = pd.read_csv('infsP.csv')

df.head()


# %%

df = df.loc[df['infusionoffset'] > 0]
df['Time'] = df['infusionoffset'].div(60)

df.head()


# %%

# PROCESS: 
# ask about the 1.8% nacl having different spacings and stuff. 
# note that nacl is all over the place, reassign all versions to one 
# group. 

# pool nacl into a separate group and parse through it later 

grouped = ['1.8%NaCl \(ml/hr\)','3% NaCl \(ml/hr\)',
           '3% saline \(ml/hr\)', 'Cisatracurium \(mcg/kg/min\)', 
           'Atracurium \(mcg/kg/min\)', 'Dexmedetomidine \(mcg/kg/hr\)',
           'Dexmedetomidine \(ml/hr\)', 'Dopamine \(mcg/kg/min\)',
            'Epinephrine \(ml/hr\)', 'EPINEPHrine\(Adrenalin\)STD 4 mg Sodium Chloride 0.9% 250 ml \(mcg/min\)', 'Epinephrine',
            'Fentanyl \(\)', 'Fentanyl \(mcg/hr\)', 'Fentanyl \(ml/hr\)','Fentanyl',
           'FentaNYL \(Sublimaze\) 2500 mcg Sodium Chloride 0.9% 250 ml  Premix \(mcg/hr\)',
            'LORazepam \(Ativan\) 100 mg Sodium Chloride 0.9% 100 ml  Premix \(mg/hr\)',
           'Lorazepam \(mg/hr\)','Lorazepam \(\)', 'Mannitol gtt\(25gms/500mL\) 25 g Dextrose 5% 500 ml \(Unknown\)',
           'Mannitol IVF Infused \(ml/hr\)', 'Midazolam \(mg/hr\)', 'Midazolam \(ml/hr\)'
            'Morphine \(\)', 'Morphine \(mg/hr\)', 'Morphine','NaCl 3% \(ml/hr\)',
            'Norepinephrine \(mcg/kg/min\)', 'Norepinephrine \(mcg/min\)',
           'Norepinephrine \(ml/hr\)', 'Norepinephrine STD 4 mg Dextrose 5% 250 ml \(mcg/min\)', 'Norepinephrine',
           'Pentobarbital \(mg/kg/hr\)',  'Phenylephrine  STD 20 mg Sodium Chloride 0.9% 250 ml \(mcg/min\)',
           'Phenylephrine \(\)', 'Phenylephrine \(mcg/kg/min\)', 'Phenylephrine \(mcg/min\)', 
           'Phenylephrine \(ml/hr\)', 'Phenylephrine',  'Vasopressin \(\)', 'Vasopressin \(ml/hr\)',
           'Vasopressin \(units/min\)', 'Vasopressin 40 Units Sodium Chloride 0.9% 100 ml \(units/kg/hr\)', 'Vasopressin',
           'vecuronium \(mcg/kg/min\)', 'Vecuronium \(Norcuron\) 100 mg Sodium Chloride 0.9% 100 ml \(mcg/kg/min\)'  
           ]


# just keeping it simpler for now
# df = df[df['drugname'].isin(grouped)]
pattern = '|'.join(grouped)
df_filtered = df[df['drugname'].str.contains(pattern, case=False, regex=True)]

df.head()


# %%
df = df.sort_values(by=['patientunitstayid', 'Time'])
df.head(100000)




