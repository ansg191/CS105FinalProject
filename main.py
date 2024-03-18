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
#import numpy as np
#import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

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

relationship_income_count = df.groupby(['relationship', 'income'], observed=True).size().unstack(fill_value=0)

relationship_income_count.plot(kind='bar', figsize=(10, 6))
plt.title('Relationship Status vs Income')
plt.xlabel('Relationship Status')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Income')
plt.tight_layout()

plt.show()

contingency_table = pd.crosstab(df['relationship'], df['income'])

contingency_table_normalized = contingency_table.div(contingency_table.sum(axis=1), axis=0)

plt.figure(figsize=(10, 6))
sns.heatmap(contingency_table_normalized, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Proportion of Income Levels by Relationship Status')
plt.xlabel('Income')
plt.ylabel('Relationship Status')

plt.show()

education_income_count = df.groupby(['education', 'income'], observed=True).size().unstack(fill_value=0)

education_income_count.plot(kind='bar', figsize=(10, 6))
plt.title('Education vs Income')
plt.xlabel('Education')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Income')
plt.tight_layout()

plt.show()

contingency_table = pd.crosstab(df['education'], df['income'])

contingency_table_normalized = contingency_table.div(contingency_table.sum(axis=1), axis=0)

plt.figure(figsize=(12, 8))
sns.heatmap(contingency_table_normalized, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Proportion of Income Levels by Education')
plt.xlabel('Income')
plt.ylabel('Education Level')
plt.xticks(rotation=45)
plt.show()

workclass_income_count = df.groupby(['workclass', 'income'], observed=True).size().unstack(fill_value=0)

workclass_income_count.plot(kind='bar', figsize=(10, 6))
plt.title('Work Class vs Income')
plt.xlabel('Work Class')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Income')
plt.tight_layout()

plt.show()

plt.figure(figsize=(8, 6))
sns.violinplot(x='income', y='age', data=df)
plt.title('Age Distribution by Income')
plt.xlabel('Income')
plt.ylabel('Age')

plt.show()
