import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def process_previous_application_features(prev):
    """
    Section 4.3 Previous Application Features
    """
    
    # Safe division
    def safe_div(a, b):
        return a / b.replace(0, np.nan)
        
    prev['APP_CREDIT_RATIO'] = safe_div(prev['AMT_APPLICATION'], prev['AMT_CREDIT'])
    prev['APPROVED'] = (prev['NAME_CONTRACT_STATUS'] == 'Approved').astype(int)
    prev['REFUSED'] = (prev['NAME_CONTRACT_STATUS'] == 'Refused').astype(int)
    
    # Get last 5 statuses
    # Sort by DAYS_DECISION descending
    prev = prev.sort_values(by=['SK_ID_CURR', 'DAYS_DECISION'], ascending=[True, False])
    
    # Lags for the last 5 apps
    # We can create columns APP_STATUS_LAG_1 .. 5
    prev['LAG_IDX'] = prev.groupby('SK_ID_CURR').cumcount() + 1
    
    lags = prev[prev['LAG_IDX'] <= 5].pivot(index='SK_ID_CURR', columns='LAG_IDX', values='NAME_CONTRACT_STATUS')
    lags.columns = [f'PREV_STATUS_LAG_{c}' for c in lags.columns]
    lags = lags.reset_index()
    
    # Most recent product combination
    latest_prod = prev[prev['LAG_IDX'] == 1][['SK_ID_CURR', 'NAME_GOODS_CATEGORY', 'NAME_YIELD_GROUP', 'NAME_PORTFOLIO']]
    # Simplify to just the raw columns to be encoded later.
    
    agg_dict = {
        'SK_ID_PREV': ['count'],
        'APPROVED': ['mean'],
        'REFUSED': ['mean'],
        'APP_CREDIT_RATIO': ['mean'],
        'AMT_ANNUITY': ['mean'],
        'AMT_CREDIT': ['mean'],
        'DAYS_DECISION': ['mean', 'min'],
        'AMT_GOODS_PRICE': ['mean']
    }
    
    prev_agg = prev.groupby('SK_ID_CURR').agg(agg_dict)
    
    new_cols = []
    for col in prev_agg.columns.values:
        new_cols.append(f"PREV_{col[0]}_{col[1].upper()}")
                
    prev_agg.columns = new_cols
    prev_agg = prev_agg.reset_index()
    
    prev_agg.rename(columns={
        'PREV_SK_ID_PREV_COUNT': 'PREV_COUNT',
        'PREV_APPROVED_MEAN': 'PREV_APPROVED_RATE',
        'PREV_REFUSED_MEAN': 'PREV_REFUSED_RATE',
        'PREV_APP_CREDIT_RATIO_MEAN': 'PREV_APP_CREDIT_RATIO_MEAN',
        'PREV_AMT_ANNUITY_MEAN': 'PREV_ANNUITY_MEAN',
        'PREV_AMT_CREDIT_MEAN': 'PREV_CREDIT_MEAN',
        'PREV_DAYS_DECISION_MEAN': 'PREV_DAYS_DECISION_MEAN',
        'PREV_DAYS_DECISION_MIN': 'PREV_DAYS_DECISION_MIN',
        'PREV_AMT_GOODS_PRICE_MEAN': 'PREV_GOODS_PRICE_MEAN'
    }, inplace=True)
    
    # Sliced aggregates for last 3/5 applications
    prev_last_3 = prev[prev['LAG_IDX'] <= 3].groupby('SK_ID_CURR')['AMT_CREDIT'].mean().reset_index().rename(columns={'AMT_CREDIT': 'PREV_CREDIT_MEAN_LAST3'})
    prev_last_5 = prev[prev['LAG_IDX'] <= 5].groupby('SK_ID_CURR')['AMT_CREDIT'].mean().reset_index().rename(columns={'AMT_CREDIT': 'PREV_CREDIT_MEAN_LAST5'})
    
    # Merge all
    df_merged = prev_agg.merge(lags, on='SK_ID_CURR', how='left')
    df_merged = df_merged.merge(latest_prod, on='SK_ID_CURR', how='left')
    df_merged = df_merged.merge(prev_last_3, on='SK_ID_CURR', how='left')
    df_merged = df_merged.merge(prev_last_5, on='SK_ID_CURR', how='left')
    
    return df_merged

if __name__ == "__main__":
    from src.data.loader import load_previous_applications
    prev = load_previous_applications()
    prev_features = process_previous_application_features(prev)
    print("Previous Apps Features Shape:", prev_features.shape)
