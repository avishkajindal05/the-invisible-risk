import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

def generate_financial_narratives(df):
    """
    Generate text describing the financial profile of the applicant
    (from DAYS_BIRTH, AMT_INCOME_TOTAL, EXT_SOURCE_MEAN, DAYS_EMPLOYED, CNT_CHILDREN, FLAG_OWN_REALTY)
    """
    narratives = []
    
    # We expect these columns to be available
    for idx, row in df.iterrows():
        age = int(abs(row.get('DAYS_BIRTH', 0)) / 365.25)
        income = row.get('AMT_INCOME_TOTAL', 0)
        ext_mean = row.get('EXT_SOURCE_MEAN', 0.5)
        emp_years = int(abs(row.get('DAYS_EMPLOYED', 0)) / 365.25)
        kids = int(row.get('CNT_CHILDREN', 0))
        realty = row.get('FLAG_OWN_REALTY', 'N')
        
        narrative = f"The applicant is {age} years old with an annual income of {income:.0f}. "
        narrative += f"They have {kids} children. "
        narrative += f"Employment history is {emp_years} years. "
        narrative += f"External credit assessment is {ext_mean:.3f}. "
        narrative += f"Real estate ownership: {realty}."
        
        narratives.append(narrative)
        
    return narratives

def process_nlp_embeddings(df):
    """
    Encode with Sentence-BERT -> 384-dim
    Reduce with PCA to 32 dimensions
    """
    print("Generating narratives...")
    narratives = generate_financial_narratives(df)
    
    print("Loading SentenceTransformer all-MiniLM-L6-v2...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Encoding embeddings...")
    embeddings = model.encode(narratives, batch_size=128, show_progress_bar=True)
    
    print("Applying PCA to reduce from 384 to 32 dimensions...")
    pca = PCA(n_components=32, random_state=42)
    embeddings_32 = pca.fit_transform(embeddings)
    
    # Generate column names
    pca_cols = [f"NLP_EMB_{i}" for i in range(32)]
    df_embeddings = pd.DataFrame(embeddings_32, columns=pca_cols)
    
    # Concatenate or merge
    # Assuming the input dataframe is aligned in row index
    for col in pca_cols:
        df[col] = df_embeddings[col].values
        
    return df

if __name__ == "__main__":
    import os
    print("This requires processed train/test data to run properly.")
    # test_df = pd.DataFrame({'DAYS_BIRTH': [-15000], 'AMT_INCOME_TOTAL': [200000], 'EXT_SOURCE_MEAN': [0.5], 'DAYS_EMPLOYED': [-2000], 'CNT_CHILDREN': [1], 'FLAG_OWN_REALTY': ['Y']})
    # processed = process_nlp_embeddings(test_df)
    # print(processed.head())
