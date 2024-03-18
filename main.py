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
