import nbformat as nbf

nb = nbf.v4.new_notebook()

text = """# Exploratory Data Analysis
This notebook performs basic Exploratory Data Analysis (EDA) on `train_processed.csv`."""
nb['cells'].append(nbf.v4.new_markdown_cell(text))

code1 = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set aesthetics for plots
sns.set_theme(style="whitegrid")
%matplotlib inline"""
nb['cells'].append(nbf.v4.new_code_cell(code1))

code2 = """# Load Data
data_path = r'../Data/processed/train_processed.csv'
df = pd.read_csv(data_path)

print(f"Dataset Shape: {df.shape}")
df.head()"""
nb['cells'].append(nbf.v4.new_code_cell(code2))

code3 = """# Basic Information
df.info()"""
nb['cells'].append(nbf.v4.new_code_cell(code3))

code4 = """# Summary Statistics
display(df.describe())"""
nb['cells'].append(nbf.v4.new_code_cell(code4))

code5 = """# Check for Missing Values
missing_vals = df.isnull().sum()
missing_vals = missing_vals[missing_vals > 0].sort_values(ascending=False)

if len(missing_vals) > 0:
    plt.figure(figsize=(10, 5))
    sns.barplot(x=missing_vals.values, y=missing_vals.index)
    plt.title('Missing Values Count')
    plt.show()
else:
    print("No missing values found!")"""
nb['cells'].append(nbf.v4.new_code_cell(code5))

code6 = """# Target Distribution
if 'TARGET' in df.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='TARGET', palette='viridis')
    plt.title('Target Variable Distribution')
    plt.show()
    
    print("Target normalized counts:\\n", df['TARGET'].value_counts(normalize=True))
else:
    print("TARGET column not found.")"""
nb['cells'].append(nbf.v4.new_code_cell(code6))

code7 = """# Correlation Heatmap for Top Features
if 'TARGET' in df.columns:
    # Get top 20 correlated numerical features with TARGET
    correlations = df.select_dtypes(include=[np.number]).corr()['TARGET'].abs().sort_values(ascending=False)
    top_features = correlations.head(21).index # Include TARGET itself
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(df[top_features].corr(), annot=False, cmap='coolwarm')
    plt.title('Correlation Heatmap of Top 20 Features with TARGET')
    plt.show()"""
nb['cells'].append(nbf.v4.new_code_cell(code7))

import os
os.makedirs('notebooks', exist_ok=True)
with open('notebooks/EDA_train_processed.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook generated successfully at notebooks/EDA_train_processed.ipynb")
