import json
import os

cells = []

def add_md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [text]})

def add_code(code_str):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\\n' for line in code_str.split('\\n')]
    })

add_md("# Advanced In-Depth Exploratory Data Analysis\\nThis notebook conducts an exhaustive EDA on `train_processed.csv`. We will explore feature distributions, bivariate relationships with the TARGET, identify patterns of variance, and detect anomalies. This gives us foundational intuition for modeling defaults.")

code1 = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from IPython.display import display

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", palette="muted")
%matplotlib inline"""
add_code(code1)

code2 = """# Load Data
data_path = r'../Data/processed/train_processed.csv'
df = pd.read_csv(data_path)

print(f"Dataset Shape: {df.shape}")
display(df.head())"""
add_code(code2)

add_md("## 1. Missing Value and DataType Diagnostic")

code3 = """# Check missing value percentage
missing_percent = (df.isnull().sum() / len(df)) * 100
missing_vals = missing_percent[missing_percent > 0].sort_values(ascending=False)

if len(missing_vals) > 0:
    plt.figure(figsize=(14, 6))
    sns.barplot(x=missing_vals.head(30).values, y=missing_vals.head(30).index, palette='Reds_r')
    plt.title('Top 30 Features with Missing Values (%)', fontsize=15)
    plt.xlabel('Percentage of Missing Values')
    plt.show()
else:
    print("No missing values found!")"""
add_code(code3)

code4 = """# Datatypes Summary
dtype_counts = df.dtypes.value_counts()
plt.figure(figsize=(8, 4))
sns.barplot(x=dtype_counts.index.astype(str), y=dtype_counts.values, palette='Blues')
plt.title('Feature DataType Distribution')
plt.ylabel('Count')
plt.show()"""
add_code(code4)

add_md("## 2. Target Variable Analysis\\nEvaluating class imbalance.")

code5 = """if 'TARGET' in df.columns:
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    # Countplot
    sns.countplot(data=df, x='TARGET', ax=ax[0], palette='viridis')
    ax[0].set_title('Target Class absolute Count')
    
    # Pie chart
    target_counts = df['TARGET'].value_counts()
    ax[1].pie(target_counts, labels=target_counts.index, autopct='%1.1f%%', colors=sns.color_palette('viridis', 2), explode=(0, 0.1))
    ax[1].set_title('Target Class Relative Proportion')
    
    plt.tight_layout()
    plt.show()"""
add_code(code5)

add_md("## 3. Top Numerical Features - KDE Distributions\\nAnalyzing how numerical features separate the TARGET space.")

code6 = """numeric_features = df.select_dtypes(include=[np.number]).columns.drop(['TARGET', 'SK_ID_CURR'], errors='ignore')

if 'TARGET' in df.columns:
    # Let's pick 6 top correlated features to visualize distributions safely
    corrs = df[numeric_features].corrwith(df['TARGET']).abs().sort_values(ascending=False)
    top_6_num = corrs.head(6).index
    
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(top_6_num, 1):
        plt.subplot(2, 3, i)
        sns.kdeplot(data=df, x=col, hue='TARGET', common_norm=False, fill=True, alpha=0.3, palette='coolwarm')
        plt.title(f'{col} Distribution by Target')
    plt.tight_layout()
    plt.show()"""
add_code(code6)

add_md("## 4. Top Categorical Features vs Target\\nFrequency of defaults across major categorical strata.")

code7 = """categorical_features = df.select_dtypes(include=['object', 'category']).columns

if len(categorical_features) > 0 and 'TARGET' in df.columns:
    # Plot top 4 categorical features with least unique values to avoid extreme clutter
    unique_counts = df[categorical_features].nunique().sort_values()
    top_cats = unique_counts[unique_counts < 10].head(4).index
    
    plt.figure(figsize=(16, 10))
    for i, col in enumerate(top_cats, 1):
        plt.subplot(2, 2, i)
        # We calculate mean target representing the "Default Rate"
        prop_df = df.groupby(col)['TARGET'].mean().reset_index()
        sns.barplot(data=prop_df, x=col, y='TARGET', palette='magma')
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Default Rate across {col}')
        plt.ylabel('Average Default %')
    plt.tight_layout()
    plt.show()"""
add_code(code7)

add_md("## 5. Global Correlation Map\\nMulticollinearity analysis using a cluster map logic for the highest variance continuous features.")

code8 = """if 'TARGET' in df.columns:
    highly_corr = corrs.head(25).index.tolist() + ['TARGET']
    corr_matrix = df[highly_corr].corr()
    
    plt.figure(figsize=(15, 13))
    # Draw lower triangle heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='vlag', vmin=-1, vmax=1, center=0, 
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=False)
    plt.title('Correlation Matrix of Top 25 Highest Correlated Features', fontsize=16)
    plt.show()"""
add_code(code8)

add_md("## 6. Outlier Diagnostics\\nUsing Boxplots against the Target.")

code9 = """if len(top_6_num) > 0:
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(top_6_num, 1):
        plt.subplot(2, 3, i)
        sns.boxplot(data=df, x='TARGET', y=col, palette='Set2')
        plt.title(f'{col} Boxplot')
    plt.tight_layout()
    plt.show()"""
add_code(code9)

add_md("### Next Steps\\nProceed to Feature Selection removing highly collinear numerical variables identified in Chapter 5, and execute Outlier smoothing techniques observed in Chapter 6 before Pipeline generation.")

nb_dict = {
 "cells": cells,
 "metadata": {
  "kernelspec": {
   "display_name": "mgenv",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {"name": "ipython", "version": 3},
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.19"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

# The user path is 'notebooks/EDA_train_processed.ipynb'
out_path = r'd:/IITGN/Project/2 - Invincibles/CodeBase/notebooks/EDA_train_processed.ipynb'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(nb_dict, f, indent=1)
print("Notebook Successfully Rewritten!")
