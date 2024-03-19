"""
CS105 Final Project
Authors:
    - Anshul Gupta <agupt109@ucr.edu>
    - Ali Naqvi <anaqv007@ucr.edu>
    - Alex Zhang <azhan061@ucr.edu>
    - Nathan Lee <nlee097@ucr.edu>

This file contains all of the python code for the project.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix

df = pd.read_csv('census_income/adult.data')

# Column names are missing from the CSV
# See `census_income/adult.names` for the column names
df.columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country', 'income',
]

# Convert categorical features
df['workclass'] = df['workclass'].astype('category')
df['marital-status'] = df['marital-status'].astype('category')
df['occupation'] = df['occupation'].astype('category')
df['relationship'] = df['relationship'].astype('category')
df['race'] = df['race'].astype('category')
df['sex'] = df['sex'].astype('category')
df['native-country'] = df['native-country'].astype('category')
df['income'] = df['income'].astype('category')

print(df)

#%%

# Plots `Hours per week worked` vs `education level` with the color representing their `income`
df.plot.scatter(x='hours-per-week', y='education-num', c='income', colormap='viridis')
plt.show()

#%%

relationship_income_count = df.groupby(['relationship', 'income'], observed=True).size().unstack(
    fill_value=0)

relationship_income_count.plot(kind='bar', figsize=(10, 6))
plt.title('Relationship Status vs Income')
plt.xlabel('Relationship Status')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Income')
plt.tight_layout()

plt.show()

#%%

contingency_table = pd.crosstab(df['relationship'], df['income'])

contingency_table_normalized = contingency_table.div(contingency_table.sum(axis=1), axis=0)

plt.figure(figsize=(10, 6))
sns.heatmap(contingency_table_normalized, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Proportion of Income Levels by Relationship Status')
plt.xlabel('Income')
plt.ylabel('Relationship Status')

plt.show()

#%%

education_income_count = df.groupby(['education', 'income'], observed=True).size().unstack(
    fill_value=0)

education_income_count.plot(kind='bar', figsize=(10, 6))
plt.title('Education vs Income')
plt.xlabel('Education')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Income')
plt.tight_layout()

plt.show()

#%%

contingency_table = pd.crosstab(df['education'], df['income'])

contingency_table_normalized = contingency_table.div(contingency_table.sum(axis=1), axis=0)

plt.figure(figsize=(12, 8))
sns.heatmap(contingency_table_normalized, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Proportion of Income Levels by Education')
plt.xlabel('Income')
plt.ylabel('Education Level')
plt.xticks(rotation=45)
plt.show()

#%%

workclass_income_count = df.groupby(['workclass', 'income'], observed=True).size().unstack(
    fill_value=0)

workclass_income_count.plot(kind='bar', figsize=(10, 6))
plt.title('Work Class vs Income')
plt.xlabel('Work Class')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Income')
plt.tight_layout()

plt.show()

#%%

plt.figure(figsize=(8, 6))
sns.violinplot(x='income', y='age', data=df)
plt.title('Age Distribution by Income')
plt.xlabel('Income')
plt.ylabel('Age')

plt.show()

#%%

X = df[['age', 'education-num', 'hours-per-week', 'capital-gain', 'capital-loss']]
y = df['income'].cat.codes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

k = 5
knn_regressor = KNeighborsRegressor(n_neighbors=k)
knn_regressor.fit(X_train_scaled, y_train)

# Predictions
y_pred = knn_regressor.predict(X_test_scaled)
y_pred = np.round(y_pred, 0)
# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)
print(confusion_matrix(y_test, y_pred))

# %%

features = ['age', 'workclass', 'education-num', 'marital-status', 'occupation',
            'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
            'native-country']
cats = [False, True, False, True, True, True, True, True, False, False, False, True]


def run_knn(X, y, k):
    """
    Runs K-Nearest Neighbors regression on X & y and calculates the MSE

    :param X: array-like of shape (n_samples, n_features)
    :param y: array-like of shape (n_samples)
    :param k: K parameter for KNN
    :return: (Mean Squared Error, R^2)
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn_regressor = KNeighborsRegressor(n_neighbors=k)
    knn_regressor.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = knn_regressor.predict(X_test_scaled)
    # y_pred = np.round(y_pred, 0)
    # Evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, r2


X = pd.DataFrame()
y = df['income'].cat.codes

feat_set = set()  # Features that have already been selected by the algorithm
results = []  # Best results for each level of the tree
for j in range(len(features)):
    print(j)  # Cause this takes a while

    # Track best MSE feature for this tree level
    best_mse = float('inf')
    best_idx = None

    for i in range(len(features)):
        feat = features[i]
        is_cat = cats[i]

        # Prevent duplicate features
        if feat in feat_set:
            continue

        # print(i, feat)

        # Add feature as column to X
        X[feat] = df[feat].cat.codes if is_cat else df[feat]

        # Run KNN on X
        mse, _ = run_knn(X, y, 15)
        if best_idx is None or mse < best_mse:
            best_idx = i
            best_mse = mse

        # Remove feature column from X
        X.drop(feat, axis=1, inplace=True)

    assert (best_idx is not None)

    # Add feature to X
    X[features[best_idx]] = df[features[best_idx]].cat.codes if cats[best_idx] else df[
        features[best_idx]]
    # Add feature to feature set
    feat_set.add(features[best_idx])

    # Add result of tree level
    results.append((X.copy(deep=True), best_mse))

# Find minimum MSE at all tree levels
min_mse = float('inf')
min_feat = pd.DataFrame()
for result in results:
    if result[1] < min_mse:
        min_mse = result[1]
        min_feat = result[0]

print(min_feat.columns)
print(min_mse)

X = min_feat

# %%
# Elbow Method for KNN

ks = np.linspace(1, 25, 13, dtype=int)
errs = np.zeros((len(ks), 2))

for i in range(13):
    k = ks[i]
    knn_regressor = KNeighborsRegressor(n_neighbors=k)
    knn_regressor.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = knn_regressor.predict(X_test_scaled)
    # Evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    errs[i] = (mse, r2)

fig, ax1 = plt.subplots()

COLOR = 'tab:red'
ax1.set_xlabel("K value")
ax1.set_ylabel("Mean Squared Error", color=COLOR)
ax1.plot(ks, errs[:, 0], color=COLOR)
ax1.tick_params(axis='y', labelcolor=COLOR)

COLOR = 'tab:blue'
ax2 = ax1.twinx()
ax2.set_ylabel("R^2 Score", color=COLOR)
ax2.plot(ks, errs[:, 1], color=COLOR)
ax2.tick_params(axis='y', labelcolor=COLOR)

fig.suptitle('Elbow Method for KNN')
fig.tight_layout()

plt.show()

#%%
