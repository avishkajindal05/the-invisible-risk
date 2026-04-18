import pandas as pd
import numpy as np
import os

DATA_DIR = r"d:\IITGN\Project\2 - Invincibles\CodeBase\Data"

def load_application_data(is_train=True):
    """
    Loads application_train.csv or application_test.csv.
    Applies initial missing value strategies based on the specification.
    """
    file_name = "application_train.csv" if is_train else "application_test.csv"
    filepath = os.path.join(DATA_DIR, file_name)
    df = pd.read_csv(filepath)
    
    # --- Missing Value Strategy & Leakage Handling ---
    
    # 1. Missing EXT_SOURCE (Impute median + create missing flags)
    # The prompt specified imputing median, but median is better computed per dataset or robustly.
    # We will compute the median over the current dataframe.
    for i in [1, 2, 3]:
        col = f'EXT_SOURCE_{i}'
        if col in df.columns:
            df[f'{col}_MISSING'] = df[col].isnull().astype(int)
            # Impute with median
            df[col] = df[col].fillna(df[col].median())
            
    # 2. Missing occupation/org type (Treat as "Unknown")
    for col in ['OCCUPATION_TYPE', 'ORGANIZATION_TYPE']:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
            
    # 3. Handle DAYS_EMPLOYED > 0 (Retired)
    if 'DAYS_EMPLOYED' in df.columns:
        df['IS_RETIRED'] = (df['DAYS_EMPLOYED'] > 0).astype(int)
        # Replace positive values (which mean retired) with 0 or nan? "replace with 0"
        df.loc[df['DAYS_EMPLOYED'] > 0, 'DAYS_EMPLOYED'] = 0
        
    # Remove any leaked features if they exist (though application table typically doesn't have target leaks)
    # Only TARGET should be retained if train
    
    return df

def load_bureau_data():
    bureau = pd.read_csv(os.path.join(DATA_DIR, "bureau.csv"))
    bureau_balance = pd.read_csv(os.path.join(DATA_DIR, "bureau_balance.csv"))
    return bureau, bureau_balance

def load_previous_applications():
    return pd.read_csv(os.path.join(DATA_DIR, "previous_application.csv"))

def load_installments_payments():
    return pd.read_csv(os.path.join(DATA_DIR, "installments_payments.csv"))

def load_pos_cash_balance():
    return pd.read_csv(os.path.join(DATA_DIR, "POS_CASH_balance.csv"))

def load_credit_card_balance():
    return pd.read_csv(os.path.join(DATA_DIR, "credit_card_balance.csv"))

if __name__ == "__main__":
    train_df = load_application_data(is_train=True)
    print("Train Shape:", train_df.shape)
    print("Retired Percentage:", train_df['IS_RETIRED'].mean())
