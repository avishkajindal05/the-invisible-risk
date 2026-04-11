import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce

def apply_categorical_encodings(train_df, test_df):
    """
    Implements Section 4.8 High-Cardinality Categorical Encoding.
    OCCUPATION_TYPE -> K-Means clustering (k=6) on (default_rate, income_mean)
    ORGANIZATION_TYPE -> Target encoding + frequency encoding
    Other Categoricals -> Label encoding
    """
    
    # 1. OCCUPATION_TYPE K-Means
    if 'TARGET' in train_df.columns:
        occ_stats = train_df.groupby('OCCUPATION_TYPE').agg(
            default_rate=('TARGET', 'mean'),
            income_mean=('AMT_INCOME_TOTAL', 'mean')
        ).reset_index()
        
        # Scale for K-Means
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(occ_stats[['default_rate', 'income_mean']])
        
        kmeans = KMeans(n_clusters=6, random_state=42)
        occ_stats['OCCUPATION_CLUSTER'] = kmeans.fit_predict(scaled_features)
        
        # Merge back
        train_df = train_df.merge(occ_stats[['OCCUPATION_TYPE', 'OCCUPATION_CLUSTER']], on='OCCUPATION_TYPE', how='left')
        test_df = test_df.merge(occ_stats[['OCCUPATION_TYPE', 'OCCUPATION_CLUSTER']], on='OCCUPATION_TYPE', how='left')
        
    # 2. ORGANIZATION_TYPE
    if 'TARGET' in train_df.columns:
        # Target Encoder
        te = ce.TargetEncoder(cols=['ORGANIZATION_TYPE'])
        train_df['ORGANIZATION_TYPE_TARGET_ENC'] = te.fit_transform(train_df['ORGANIZATION_TYPE'], train_df['TARGET'])
        test_df['ORGANIZATION_TYPE_TARGET_ENC'] = te.transform(test_df['ORGANIZATION_TYPE'])
        
        # Freq Encoder
        freq = train_df['ORGANIZATION_TYPE'].value_counts(normalize=True)
        train_df['ORGANIZATION_TYPE_FREQ_ENC'] = train_df['ORGANIZATION_TYPE'].map(freq)
        test_df['ORGANIZATION_TYPE_FREQ_ENC'] = test_df['ORGANIZATION_TYPE'].map(freq)
        
    # 3. Handle object columns (Label Encode)
    cat_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    
    le = LabelEncoder()
    for col in cat_cols:
        # Fill missing before label encoding, to keep sizes matched
        train_df[col] = train_df[col].fillna('Unknown')
        test_df[col] = test_df[col].fillna('Unknown')
        
        le.fit(train_df[col].tolist() + test_df[col].tolist())
        train_df[col] = le.transform(train_df[col])
        test_df[col] = le.transform(test_df[col])
        
    return train_df, test_df

if __name__ == "__main__":
    print("Categorical encodings module ready.")
