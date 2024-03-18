"""
CS105 Final Project
Authors:
    - Anshul Gupta <agupt109@ucr.edu>
    - Ali Naqvi <anaqv007@ucr.edu>
    - Alex Zhang <azhan061@ucr.edu>
    - Nathan Lee <nlee097@ucr.edu>

This file contains all of the python code for the project.
"""

import pandas as pd
# import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

df = pd.read_csv('census_income/adult.data')

# Column names are missing from the CSV
# See `census_income/adult.names` for the column names
df.columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country', 'income',
]

# Convert categorical features
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'income']
df_encoded = pd.get_dummies(df, columns=categorical_cols)

X = df[['age', 'education-num', 'hours-per-week', 'capital-gain', 'capital-loss', 'income']]
X_encoded = pd.get_dummies(X, columns=['income'])

inertia_values = []

for k in range(1, 26):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_encoded)
    inertia_values.append(kmeans.inertia_)

# Plot the results
plt.plot(range(1, 26), inertia_values, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

#kmeans = KMeans(n_clusters=3, random_state=42)
#kmeans.fit(df_encoded)

#cluster_centers = kmeans.cluster_centers_
#cluster_labels = kmeans.labels_

#df['cluster_label'] = cluster_labels

#print("Cluster Centers:")
#print(cluster_centers)

#print("\nCounts of samples in each cluster:")
#print(df['cluster_label'].value_counts())