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
# import matplotlib.pyplot as plt
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

print(df_encoded)
print(df_encoded.dtypes)

# Initialize and fit KMeans model
kmeans = KMeans(n_clusters=3, random_state=42)  # Setting number of clusters to 3, you can change this
kmeans.fit(df_encoded)

# Get cluster centers
cluster_centers = kmeans.cluster_centers_

# Get cluster labels
cluster_labels = kmeans.labels_

# Add cluster labels to original dataframe
df['cluster_label'] = cluster_labels

# Print cluster centers
print("Cluster Centers:")
print(cluster_centers)

# Print counts of samples in each cluster
print("\nCounts of samples in each cluster:")
print(df['cluster_label'].value_counts())