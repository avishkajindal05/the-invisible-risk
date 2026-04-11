import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def process_bureau_features(bureau, bureau_balance):
    """
    Implements Section 4.2 Bureau Features (~35 features after aggregation).
    """
    
    # 1. Process bureau_balance.csv
    # Map STATUS: C, X -> 0 (Closed/Unknown but safe). 0 -> 0. 1..5 are integers 1..5.
    bb = bureau_balance.copy()
    bb['STATUS'] = bb['STATUS'].replace({'C': 0, 'X': 0})
    bb['STATUS'] = pd.to_numeric(bb['STATUS'])
    
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(
        BUREAU_STATUS_WORST_PER_BUREAU=('STATUS', 'max'),
        BUREAU_STATUS_MEAN_PER_BUREAU=('STATUS', 'mean')
    ).reset_index()
    
    # Merge back to bureau
    bureau = bureau.merge(bb_agg, on='SK_ID_BUREAU', how='left')
    
    # Safe division
    def safe_div(a, b):
        return a / b.replace(0, np.nan)
        
    bureau['DEBT_CREDIT_RATIO'] = safe_div(bureau['AMT_CREDIT_SUM_DEBT'], bureau['AMT_CREDIT_SUM'])
    bureau['CREDIT_UTIL_RATE'] = safe_div(bureau['AMT_CREDIT_SUM_OVERDUE'], bureau['AMT_CREDIT_SUM'])
    bureau['CREDIT_ACTIVE_BINARY'] = (bureau['CREDIT_ACTIVE'] == 'Active').astype(int)
    
    # Only active credits for LAST_ACTIVE_DAYS_CREDIT
    active_bureau = bureau[bureau['CREDIT_ACTIVE'] == 'Active']
    last_active = active_bureau.groupby('SK_ID_CURR')['DAYS_CREDIT'].max().reset_index()
    last_active.rename(columns={'DAYS_CREDIT': 'LAST_ACTIVE_DAYS_CREDIT'}, inplace=True)
    
    # Main aggregation dictionary
    agg_dict = {
        'DAYS_CREDIT': ['mean', 'min', 'max'],
        'CREDIT_DAY_OVERDUE': ['mean', 'max'],
        'AMT_CREDIT_SUM': ['mean', 'max', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['mean', 'max'],
        'DEBT_CREDIT_RATIO': ['mean', 'max'],
        'CREDIT_UTIL_RATE': ['mean', 'max'],
        'CREDIT_ACTIVE_BINARY': ['mean', 'sum'],
        'BUREAU_STATUS_WORST_PER_BUREAU': ['mean', 'max'],
        'BUREAU_STATUS_MEAN_PER_BUREAU': ['mean'],
        'SK_ID_BUREAU': ['count']
    }
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg(agg_dict)
    
    # Flatten multi-level columns
    new_cols = []
    for col in bureau_agg.columns.values:
        new_cols.append(f"BUREAU_{col[0]}_{col[1].upper()}")
                
    bureau_agg.columns = new_cols
    bureau_agg = bureau_agg.reset_index()
    
    # Rename SK_ID_BUREAU_COUNT to BUREAU_COUNT and BUREAU_CREDIT_ACTIVE_BINARY_SUM to BUREAU_ACTIVE_COUNT
    bureau_agg.rename(columns={
        'BUREAU_SK_ID_BUREAU_COUNT': 'BUREAU_COUNT',
        'BUREAU_CREDIT_ACTIVE_BINARY_SUM': 'BUREAU_ACTIVE_COUNT',
        'BUREAU_BUREAU_STATUS_WORST_PER_BUREAU_MEAN': 'BUREAU_STATUS_WORST_MEAN',
        'BUREAU_BUREAU_STATUS_WORST_PER_BUREAU_MAX': 'BUREAU_STATUS_WORST_MAX',
        'BUREAU_BUREAU_STATUS_MEAN_PER_BUREAU_MEAN': 'BUREAU_STATUS_MEAN_MEAN'
    }, inplace=True)
    
    bureau_agg = bureau_agg.merge(last_active, on='SK_ID_CURR', how='left')
    
    return bureau_agg

if __name__ == "__main__":
    from src.data.loader import load_bureau_data
    bureau, bureau_balance = load_bureau_data()
    bureau_features = process_bureau_features(bureau, bureau_balance)
    print("Bureau Features Shape:", bureau_features.shape)
