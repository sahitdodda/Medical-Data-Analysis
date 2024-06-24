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

# use search for the main word, if it has that word exactly then check the row
# then check if there is a percent. put it at the beginning of the new word. 
# Check for the word. append that next. 
# check for the units. append last. 


# Define a function to sort words in a string
def sort_words(text):
    # Split the string into words, sort them, and then join them back into a string
    sorted_words = ' '.join(sorted(text.split()))
    return sorted_words

# Apply the sorting function to the 'text' column
df['drugname'] = df['drugname'].apply(sort_words)

# View the DataFrame
df.head()


# %%

# grouped = '1.8% NaCl (ml/hr) 3% NaCl (ml/hr) 3% saline (ml/hr) Atracurium (mcg/kg/min) Cisatracurium (mcg/kg/min) Dexmedetomidine (mcg/kg/hr) Dexmedetomidine (ml/hr) Dopamine (mcg/kg/min) Epinephrine Epinephrine (ml/hr) EPINEPHrine (Adrenalin) STD 4 mg Sodium Chloride 0.9% 250 ml (mcg/min) Fentanyl Fentanyl () Fentanyl (mcg/hr) Fentanyl (ml/hr) FentaNYL (Sublimaze) 2500 mcg Sodium Chloride 0.9% 250 ml Premix (mcg/hr) Lorazepam () LORazepam (Ativan) 100 mg Sodium Chloride 0.9% 100 ml Premix (mg/hr) Lorazepam (mg/hr) Mannitol gtt (25gms/500mL) 25 g Dextrose 5% 500 ml (Unknown) Mannitol IVF Infused (ml/hr) Midazolam (mg/hr) Midazolam (ml/hr) Morphine Morphine () Morphine (mg/hr) NaCl 3% (ml/hr) Norepinephrine Norepinephrine (mcg/kg/min) Norepinephrine (mcg/min) Norepinephrine (ml/hr) Norepinephrine STD 4 mg Dextrose 5% 250 ml (mcg/min) Pentobarbital (mg/kg/hr) Phenylephrine Phenylephrine STD 20 mg Sodium Chloride 0.9% 250 ml (mcg/min) Phenylephrine () Phenylephrine (mcg/kg/min) Phenylephrine (mcg/min) Phenylephrine (ml/hr) Vasopressin Vasopressin () Vasopressin (ml/hr) Vasopressin (units/min) Vasopressin 40 Units Sodium Chloride 0.9% 100 ml (units/kg/hr) vecuronium (mcg/kg/min) Vecuronium (Norcuron) 100 mg Sodium Chloride 0.9% 100 ml (mcg/kg/min)'

# # Define a function to sort words in a string
# def sort_words_without_units(text): 
#     # Remove numbers, percents, and unit symbols 
#     text = re.sub(r'[\d\%\\/\.]', '', text) 
#     return text

# grouped = sort_words_without_units(grouped)
# print(grouped)


# %%

# def filter_drug(item): 
#     if re.search(item, grouped):
#         return item 
#     return 'false category'

# df['drugname'] = df['drugname'].apply(filter_drug)

# df.head()


#%%


