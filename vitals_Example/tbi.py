# %%
# Imports and setting df
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from sklearn.cluster import KMeans

df = pd.read_csv('vitalsP.csv')


df.head()

# %%
# data
df.describe()

# %%
df.loc[df["cvp"] > 35, 'cvp'] = np.NaN
df.loc[df["cvp"] < -5, 'cvp'] = np.NaN
# data post cvp clean
df.describe()

# %%
# NAN map
sb.heatmap(pd.isnull(df), cmap = 'viridis')

# %%
# Total na values per column in df
df.isna().sum()

# %%
# total na count
df.isna().sum().sum()

# %%
for col in df.columns: 
    df = df.dropna(subset=[col])

df.drop_duplicates()



for col in df.columns: 
    if col != 'temperature':
        df[col] = df[col].astype(int)
    else:
        df[col] = df[col].round(2)

# %%
# combine days and hours into one column 

df["dayhour"] = df['Day'].astype(str) + "," + df['Hour'].astype(str)
print(df["dayhour"])

# %%
df = df.drop(columns=['Hour', 'Day'])

# %%
print(df.columns.tolist())

# %%


# %%
# use this as a breakpoint when generating for xgboost 
# now, try to find RMSE between each category while predicting ICP  using xgboost. export the dataframe into a new csv

df.to_csv('xgB.csv', sep = ',', index=False)




# %%
sb.scatterplot(x= df['dayhour'], y=df["icp"])

# %%
sb.lineplot(x= df['dayhour'], y=df["icp"])

# %%
sb.scatterplot(x= df['temperature'], y=df["icp"])


# %%
sb.scatterplot(x= df['sao2'], y=df["icp"])


# %%
sb.scatterplot(x= df['heartrate'], y=df["icp"])


# %%
sb.scatterplot(x= df['respiration'], y=df["icp"])


# %%
sb.scatterplot(x= df['cvp'], y=df["icp"])


# %%
sb.scatterplot(x= df['etco2'], y=df["icp"])


# %%
sb.scatterplot(x= df['systemicsystolic'], y=df["icp"])


# %%
sb.scatterplot(x= df['systemicdiastolic'], y=df["icp"])


# %%
sb.scatterplot(x= df['systemicmean'], y=df["icp"])


# %%
#New stuff added here

# %%
#all scatter plots
cols = ['patientunitstayid', 'temperature', 'sao2', 'heartrate', 'respiration', 
        'cvp', 'etco2', 'systemicsystolic', 'systemicdiastolic', 'systemicmean', 'icp']
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(cols):
    sb.scatterplot(data=df, x='dayhour', y=col, ax=axes[i])
    axes[i].set_title(col)

plt.tight_layout()
plt.show()

# %%
#group scatter plot
numeric_cols = ['temperature', 'sao2', 'heartrate', 'respiration', 'cvp', 'etco2', 'systemicsystolic', 'systemicdiastolic', 'systemicmean', 'icp']
colors = sb.color_palette("husl", len(numeric_cols))  # Generate a palette of colors

# Create a figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each numeric column as a scatter plot with a different color
for i, col in enumerate(numeric_cols):
    sb.scatterplot(data=df, x='dayhour', y=col, ax=ax, color=colors[i], label=col)

# Add a legend
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

# Set the plot title and axis labels
ax.set_title('Scatter Plots for Numeric Columns')
ax.set_xlabel('dayhour')
ax.set_ylabel('Value')

# Show the plot
plt.show()

# do this for each of the 54 patients

# %%
numeric_cols = ['temperature', 'sao2', 'heartrate', 'respiration', 'cvp', 'etco2', 'systemicsystolic', 'systemicdiastolic', 'systemicmean']
colors = sb.color_palette("husl", len(numeric_cols))  # Generate a palette of colors

# Create a figure with subplots
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 12))
axes = axes.flatten()

# Plot each numeric column as a scatter plot with a different color
for i, col in enumerate(numeric_cols):
    sb.scatterplot(data=df, x=col, y='icp', ax=axes[i], color=colors[i], label=col)
    axes[i].set_title(f"{col} vs ICP")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('ICP')
    axes[i].legend()

# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.5)

# Show the plot
plt.show()

# %%


# %%
numeric_cols = ['temperature', 'sao2', 'heartrate', 'respiration', 'cvp', 'etco2', 'systemicsystolic', 'systemicdiastolic', 'systemicmean', 'dayhour']
colors = sb.color_palette("husl", len(numeric_cols))  # Generate a palette of colors

# Create a figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each numeric column as a scatter plot with a different color
for i, col in enumerate(numeric_cols):
    sb.scatterplot(data=df, x=col, y='icp', ax=ax, color=colors[i], label=col)

# Add a legend
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

# Set the plot title and axis labels
ax.set_title('Scatter Plots of Numeric Columns vs ICP')
ax.set_xlabel('All Measurements')
ax.set_ylabel('ICP')

# Show the plot
plt.show()

# do this for each of the 54 patients

# %%
#all line plots
cols = ['patientunitstayid', 'temperature', 'sao2', 'heartrate', 'respiration', 
        'cvp', 'etco2', 'systemicsystolic', 'systemicdiastolic', 'systemicmean', 'icp']
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(cols):
    sb.lineplot(data=df, x='dayhour', y=col, ax=axes[i])
    axes[i].set_title(col)

plt.tight_layout()
plt.show()

# %%
numeric_cols = ['temperature', 'sao2', 'heartrate', 'respiration', 'cvp', 'etco2', 'systemicsystolic', 'systemicdiastolic', 'systemicmean']
colors = sb.color_palette("husl", len(numeric_cols))  # Generate a palette of colors

# Create a figure with subplots
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 12))
axes = axes.flatten()

# Plot each numeric column as a line plot with a different color
for i, col in enumerate(numeric_cols):
    sb.lineplot(data=df, x=col, y='icp', ax=axes[i], color=colors[i], label=col)
    axes[i].set_title(f"ICP vs {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('ICP')
    axes[i].legend()

# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.5)

# Show the plot
plt.show()

# %%
numeric_cols = ['temperature', 'sao2', 'heartrate', 'respiration', 'cvp', 'etco2', 'systemicsystolic', 'systemicdiastolic', 'systemicmean']
colors = sb.color_palette("husl", len(numeric_cols))  # Generate a palette of colors

# Create a figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each numeric column as a line plot with a different color
for i, col in enumerate(numeric_cols):
    sb.lineplot(data=df, x=col, y='icp', ax=ax, color=colors[i], label=col)

# Add a legend
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

# Set the plot title and axis labels
ax.set_title('Line Plots of Numeric Columns vs ICP')
ax.set_xlabel('Numeric Columns')
ax.set_ylabel('ICP')

# Show the plot
plt.show()


# %%
#group line plot
numeric_cols = ['temperature', 'sao2', 'heartrate', 'respiration', 'cvp', 'etco2', 'systemicsystolic', 'systemicdiastolic', 'systemicmean', 'icp']
colors = sb.color_palette("husl", len(numeric_cols))  # Generate a palette of colors

# Create a figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each numeric column as a line with a different color
for i, col in enumerate(numeric_cols):
    sb.lineplot(data=df, x='dayhour', y=col, ax=ax, color=colors[i], label=col)

# Add a legend
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

# Set the plot title and axis labels
ax.set_title('Line Plots for Numeric Columns')
ax.set_xlabel('dayhour')
ax.set_ylabel('Value')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Show the plot
plt.show()

# %%
# All KDE plots
numeric_cols = ['temperature', 'sao2', 'heartrate', 'respiration', 'cvp', 'etco2', 'systemicsystolic', 'systemicdiastolic', 'systemicmean', 'icp']

# Create a grid of subplots
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))
axes = axes.flatten()

# Plot each numeric column as a KDE
for i, col in enumerate(numeric_cols):
    sb.kdeplot(data=df, x=col, ax=axes[i], shade=True)
    axes[i].set_title(col)

# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()

# %%
numeric_cols = ['temperature', 'sao2', 'heartrate', 'respiration', 'cvp', 'etco2', 'systemicsystolic', 'systemicdiastolic', 'systemicmean']

# Create a grid of subplots
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
axes = axes.flatten()

# Plot each numeric column as a KDE against 'icp'
for i, col in enumerate(numeric_cols):
    sb.kdeplot(data=df, x='icp', hue=col, ax=axes[i], shade=True, legend=False)
    axes[i].set_title(f"ICP vs {col}")

# Adjust spacing between subplots
plt.tight_layout()

# mad ep
# Show the plot
plt.show()

# %%
numeric_cols = ['temperature', 'sao2', 'heartrate', 'respiration', 'cvp', 'etco2', 'systemicsystolic', 'systemicdiastolic', 'systemicmean']

# Create a figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the KDE for 'icp' colored by each numeric column
for col in numeric_cols:
    sb.kdeplot(data=df, x='icp', hue=col, ax=ax, shade=True, multiple="stack")

# Set the plot title and axis labels
ax.set_title('KDE Plot of ICP colored by Numeric Columns')
ax.set_xlabel('ICP')
ax.set_ylabel('Density')

# Add a legend
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

# Show the plot
plt.show()

# %%
# Group KDE plot
numeric_cols = ['temperature', 'sao2', 'heartrate', 'respiration', 'cvp', 'etco2', 'systemicsystolic', 'systemicdiastolic', 'systemicmean', 'icp']
colors = sb.color_palette("husl", len(numeric_cols))  # Generate a palette of colors

# Create a figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each numeric column as a KDE curve with a different color
for i, col in enumerate(numeric_cols):
    sb.kdeplot(data=df[col], ax=ax, color=colors[i], label=col, shade=True)

# Add a legend
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

# Set the plot title and axis labels
ax.set_title('KDE Plots for Numeric Columns')
ax.set_xlabel('Value')
ax.set_ylabel('Density')

# Show the plot
plt.show()

# %%


# Assuming df is your DataFrame and 'icp' is the target column

# Define the numeric columns to be used
numeric_cols = ['temperature', 'sao2', 'heartrate', 'respiration', 'cvp', 'etco2', 'systemicsystolic', 'systemicdiastolic', 'systemicmean']
colors = sns.color_palette("husl", len(numeric_cols))  # Generate a palette of colors

# Create a figure with subplots
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 12))
axes = axes.flatten()

# Number of clusters
k = 3

# Iterate over each numeric column to create scatter plots with KMeans clustering
for i, col in enumerate(numeric_cols):
    # Prepare data for KMeans
    X = df[[col, 'icp']].dropna()  # Drop NaNs to avoid issues with KMeans
    
    # Initialize and fit KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    y_pred = kmeans.fit_predict(X)
    
    # Plot the scatter plot with cluster labels
    sns.scatterplot(x=X[col], y=X['icp'], hue=y_pred, ax=axes[i], palette='viridis', legend=None)
    axes[i].set_title(f"{col} vs ICP")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('ICP')

# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.5)

# Show the plot
plt.show()


# %%
numeric_cols = ['temperature', 'sao2', 'heartrate', 'respiration', 'cvp', 'etco2', 'systemicsystolic', 'systemicdiastolic', 'systemicmean']
colors = sns.color_palette("husl", len(numeric_cols))  # Generate a palette of colors

# Create a figure with subplots
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 12))
axes = axes.flatten()

# Number of clusters
k = 3

# Iterate over each numeric column to create scatter plots with Hierarchical clustering
for i, col in enumerate(numeric_cols):
    # Prepare data for clustering
    X = df[[col, 'icp']].dropna()  # Drop NaNs to avoid issues with clustering
    
    # Initialize and fit Agglomerative Clustering
    clustering = AgglomerativeClustering(n_clusters=k)
    y_pred = clustering.fit_predict(X)
    
    # Plot the scatter plot with cluster labels
    sns.scatterplot(x=X[col], y=X['icp'], hue=y_pred, ax=axes[i], palette='viridis', legend=None)
    axes[i].set_title(f"{col} vs ICP")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('ICP')

# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.5)

# Show the plot
plt.show()


# %%
cols = ['patientunitstayid', 'temperature', 'sao2', 'heartrate', 'respiration', 
        'cvp', 'etco2', 'systemicsystolic', 'systemicdiastolic', 'systemicmean', 'icp']

# Create a figure with subplots
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 15))
axes = axes.flatten()

# Number of clusters
k = 3

# Prepare data for clustering
X = df[cols].dropna()  # Drop NaNs to avoid issues with clustering

# Initialize and fit KMeans
kmeans = KMeans(n_clusters=k, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Iterate over each column to create line plots
for i, col in enumerate(cols):
    sns.lineplot(data=df, x='dayhour', y=col, hue='cluster', ax=axes[i], palette='viridis')
    axes[i].set_title(col)

plt.tight_layout()
plt.show()


# %%
cols = ['patientunitstayid', 'temperature', 'sao2', 'heartrate', 'respiration', 
        'cvp', 'etco2', 'systemicsystolic', 'systemicdiastolic', 'systemicmean', 'icp']

# Create a figure with subplots
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 15))
axes = axes.flatten()

# Number of clusters
k = 3

# Prepare data for clustering
X = df[cols].dropna()  # Drop NaNs to avoid issues with clustering

# Initialize and fit KMeans
kmeans = KMeans(n_clusters=k, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Iterate over each column to create scatter plots
for i, col in enumerate(cols):
    sns.scatterplot(data=df, x='dayhour', y=col, hue='cluster', ax=axes[i], palette='viridis')
    axes[i].set_title(col)

plt.tight_layout()
#plt.show()
print(kmeans.cluster_centers_)

