import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def process_application_features(df):
    """
    Implements Section 4.1 Feature Engineering Specification for Application Table.
    """
    
    # Safe division helper
    def safe_div(a, b):
        return a / b.replace(0, np.nan)
        
    df['CREDIT_INCOME_RATIO'] = safe_div(df['AMT_CREDIT'], df['AMT_INCOME_TOTAL'])
    df['ANNUITY_INCOME_RATIO'] = safe_div(df['AMT_ANNUITY'], df['AMT_INCOME_TOTAL'])
    df['CREDIT_TERM'] = safe_div(df['AMT_ANNUITY'], df['AMT_CREDIT'])
    df['GOODS_CREDIT_RATIO'] = safe_div(df['AMT_GOODS_PRICE'], df['AMT_CREDIT'])
    
    df['AGE_YEARS'] = np.abs(df['DAYS_BIRTH']) / 365.25
    df['AGE_INT'] = (np.abs(df['DAYS_BIRTH']) // 365).astype(int)
    
    # DAYS_EMPLOYED has been corrected such that >0 is 0 and IS_RETIRED=1
    df['EMPLOYMENT_YEARS'] = np.abs(df['DAYS_EMPLOYED']) / 365.25
    df['EMPLOYED_RATIO'] = safe_div(df['EMPLOYMENT_YEARS'], df['AGE_YEARS'])
    
    df['CREDIT_TO_AGE'] = safe_div(df['AMT_CREDIT'], df['AGE_YEARS'])
    
    # Avoid div by zero in household/family members
    fam_members = df['CNT_FAM_MEMBERS'].fillna(1)
    fam_members = fam_members.replace(0, 1)
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / fam_members
    df['CHILDREN_RATIO'] = df['CNT_CHILDREN'] / fam_members
    
    # External Sources
    ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    df['EXT_SOURCE_MEAN'] = df[ext_cols].mean(axis=1)
    df['EXT_SOURCE_MIN'] = df[ext_cols].min(axis=1)
    df['EXT_SOURCE_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['EXT_SOURCE_STD'] = df[ext_cols].std(axis=1)
    
    df['EXT1_EXT2_INTERACTION'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2']
    df['EXT2_EXT3_INTERACTION'] = df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    
    df['EXT_CREDIT_RATIO'] = safe_div(df['EXT_SOURCE_MEAN'], df['CREDIT_INCOME_RATIO'])
    
    # Document Count (sum of FLAG_DOCUMENT_X)
    doc_cols = [c for c in df.columns if 'FLAG_DOCUMENT' in c]
    df['DOCUMENT_COUNT'] = df[doc_cols].sum(axis=1)
    
    # Enquiries
    req_cols = [c for c in df.columns if 'AMT_REQ_CREDIT_BUREAU' in c]
    df['TOTAL_ENQUIRIES'] = df[req_cols].sum(axis=1)
    if 'AMT_REQ_CREDIT_BUREAU_WEEK' in df.columns:
        df['RECENT_ENQUIRY_RATIO'] = safe_div(df['AMT_REQ_CREDIT_BUREAU_WEEK'], df['TOTAL_ENQUIRIES'])
        
    # Real Estate & Car
    if 'FLAG_OWN_CAR' in df.columns and 'FLAG_OWN_REALTY' in df.columns:
        df['HAS_CAR_REALTY'] = ((df['FLAG_OWN_CAR'] == 'Y') & (df['FLAG_OWN_REALTY'] == 'Y')).astype(int)
        
    # Address stability
    df['DAYS_REGISTRATION_RATIO'] = safe_div(df['DAYS_REGISTRATION'], df['DAYS_BIRTH'])
    
    # Winners Features
    df['CREDIT_ANNUITY_RATIO'] = safe_div(df['AMT_CREDIT'], df['AMT_ANNUITY'])
    df['CREDIT_GOODS_PRICE_RATIO'] = safe_div(df['AMT_CREDIT'], df['AMT_GOODS_PRICE'])
    
    df['CREDIT_DOWNPAYMENT'] = df['AMT_GOODS_PRICE'] - df['AMT_CREDIT']
    
    return df

if __name__ == "__main__":
    from src.data.loader import load_application_data
    df = load_application_data()
    df = process_application_features(df)
    print("Application Features Shape after Eng:", df.shape)
