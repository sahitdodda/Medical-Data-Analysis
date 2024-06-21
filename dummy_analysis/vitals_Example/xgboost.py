# %%
# Imports and setting df
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

#why does copying path not work anymore lol its fine tho 
df = pd.read_csv('xgB.csv')


df.head()

# %%
#check all dtypes, dayhour was object and xgboost doesn't like that so lets make 
# it a category
df['dayhour'] = df['dayhour'].astype('category')
df['patientunitstayid'] = df['patientunitstayid'].astype('category')

print(df.dtypes)

# %%
from sklearn.model_selection import train_test_split

# Extract feature and target arrays
X, y = df.drop('icp', axis=1), df[['icp']]

# %%
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# %%
import xgboost as xgb

# Create regression matrices
dtrain_reg = xgb.DMatrix(X_train, y_train, enable_categorical=True)
dtest_reg = xgb.DMatrix(X_test, y_test, enable_categorical=True)

# %%
params = {"objective": "reg:squarederror", "tree_method": "gpu_hist"}
n = 10000

evals = [(dtest_reg, "validation"), (dtrain_reg, "train")]


model = xgb.train(
   params=params,
   dtrain=dtrain_reg,
   num_boost_round=n,
   evals=evals,
   early_stopping_rounds=500,
   verbose_eval=50
)

# %%
# cross validation step 

params = {"objective": "reg:squarederror", "tree_method": "gpu_hist"}
n = 1000

results = xgb.cv(
   params, dtrain_reg,
   num_boost_round=n,
   nfold=5,
   early_stopping_rounds=20
)


# %%
results.head()

# %%
best_rmse = results['test-rmse-mean'].min()

best_rmse

# %%
import matplotlib.pyplot as plt

# both need to be decreasing and kind of level off 



# Create a plot with a size
plt.figure(figsize=(10, 6))

# Plotting both training and testing RMSE
plt.plot(results['train-rmse-mean'], label='Train RMSE')
plt.plot(results['test-rmse-mean'], label='Test RMSE')

# Adding title and labels
plt.title('RMSE over Boosting Rounds')
plt.xlabel('Boosting Round')
plt.ylabel('Root Mean Square Error (RMSE)')

# Adding legend
plt.legend()

# Show the plot
plt.show()

# %%
results.keys()

# %%
# the rest is kind of bs but it does let us figure out the weights

import matplotlib.pyplot as plt

# Train your XGBoost model
# Assuming you have your dmatrix objects for training data and labels
# Let's call them dtrain
xg_reg = xgb.train(params, dtrain_reg, num_boost_round=10000)

importance_type = 'weight'  # Can be 'weight', 'gain', or 'cover'
feature_importances = xg_reg.get_score(importance_type=importance_type)




# %%
print(feature_importances)

# %% 
total_importances = sum(feature_importances.values())

# for feature, importance in 
# print(feature_importances / total_importances * 100)
print(total_importances)

for feature, importance in feature_importances.items(): 
    importance_percentage = (importance / total_importances) * 100
    print(f"{feature}: {importance_percentage:.2f}%") 

# %%
import matplotlib.pyplot as plt

features, scores = zip(*feature_importances.items())
plt.bar(range(len(features)), scores, tick_label=features)
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.title('Feature Importances')
plt.show()

# %%



