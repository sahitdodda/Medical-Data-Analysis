
#%% 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Load and preprocess the data
df = pd.read_csv('examP_results.csv')
df = df.dropna()
# df = df.drop(columns=['Unnamed: 0', 'infusionoffset'])
df['physicalexamtext'] = df['physicalexamtext'].astype(str)  # Ensure the column is treated as a string

# Replace 'scored' with 0 in the 'physicalexamtext' column
df['physicalexamtext'] = df['physicalexamtext'].replace('scored', '0')

# Convert 'physicalexamtext' to numeric
df['physicalexamtext'] = pd.to_numeric(df['physicalexamtext'], errors='coerce')

# Drop any rows with NaN values after conversion
df = df.dropna()

# Normalize 'physicalexamtext' for each 'physicalexampath'
scaler = MinMaxScaler()
pathList = df['physicalexampath'].unique()

for path in pathList:
    path_df = df[df['physicalexampath'] == path].copy()
    path_df['physicalexamtext'] = scaler.fit_transform(path_df[['physicalexamtext']])
    df.loc[df['physicalexampath'] == path, 'physicalexamtext'] = path_df['physicalexamtext']

# Unique patient IDs
unique_patients = df['patientunitstayid'].unique()

# Create a figure with subplots for each patient
n_patients = len(unique_patients)
ncols = 3  # Number of columns in the grid
nrows = (n_patients // ncols) + (n_patients % ncols > 0)  # Number of rows needed

# Increase the width of the figure
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(40, 5 * nrows), sharex=True, sharey=True)

# Flatten axes array for easy iteration
axes = axes.flatten()

for i, patient_id in enumerate(unique_patients):
    ax = axes[i]
    
    # Filter data for the current patient
    patient_df = df[df['patientunitstayid'] == patient_id].copy()  # Use .copy() to avoid warnings

    # Create the line plot for the current patient
    sns.scatterplot(data=patient_df, x='Time', y='physicalexamtext', hue='physicalexampath', palette='Set2', ax=ax, s=200, markers=True)

    # Set titles and labels
    ax.set_title(f'Patient {patient_id}', fontsize=14)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Normalized Physical Exam Text', fontsize=12)

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='Exam Path', bbox_to_anchor=(1.05, 1), loc='upper left')

# Remove any empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Adjust layout to make room for legends
plt.tight_layout(rect=[0, 0, 0.8, 1])  # Adjust the right side to make room for the legend
plt.show()
