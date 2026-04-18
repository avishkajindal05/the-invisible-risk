import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def process_pos_cash_features(pos):
    """
    Section 4.5 POS Cash Features
    """
    pos['HAS_DPD'] = (pos['SK_DPD'] > 0).astype(int)
    
    agg_dict = {
        'SK_ID_PREV': ['count'],
        'SK_DPD': ['mean', 'max'],
        'HAS_DPD': ['mean'],
        'CNT_INSTALMENT': ['mean']
    }
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(agg_dict)
    
    new_cols = []
    for col in pos_agg.columns.values:
        new_cols.append(f"POS_{col[0]}_{col[1].upper()}")
                
    pos_agg.columns = new_cols
    pos_agg = pos_agg.reset_index()
    
    pos_agg.rename(columns={
        'POS_SK_ID_PREV_COUNT': 'POS_MONTHS_COUNT',
        'POS_HAS_DPD_MEAN': 'POS_DPD_RATE',
        'POS_CNT_INSTALMENT_MEAN': 'POS_CNT_INSTALMENT_MEAN'
    }, inplace=True)
    
    return pos_agg

def process_credit_card_features(cc):
    """
    Section 4.6 Credit Card Features
    """
    def safe_div(a, b):
        return a / b.replace(0, np.nan)
        
    cc['CC_UTIL_RATE'] = safe_div(cc['AMT_BALANCE'], cc['AMT_CREDIT_LIMIT_ACTUAL'])
    cc['CC_DRAWING_RATE'] = safe_div(cc['AMT_DRAWINGS_CURRENT'], cc['AMT_CREDIT_LIMIT_ACTUAL'])
    
    agg_dict = {
        'SK_ID_PREV': ['count'],
        'CC_UTIL_RATE': ['mean', 'max'],
        'CC_DRAWING_RATE': ['mean'],
        'AMT_BALANCE': ['mean'],
        'SK_DPD': ['mean']
    }
    
    cc_agg = cc.groupby('SK_ID_CURR').agg(agg_dict)
    
    new_cols = []
    for col in cc_agg.columns.values:
        new_cols.append(f"CC_{col[0]}_{col[1].upper()}")
                
    cc_agg.columns = new_cols
    cc_agg = cc_agg.reset_index()
    
    cc_agg.rename(columns={
        'CC_SK_ID_PREV_COUNT': 'CC_COUNT',
        'CC_SK_DPD_MEAN': 'CC_DPD_MEAN'
    }, inplace=True)
    
    return cc_agg

if __name__ == "__main__":
    from src.data.loader import load_pos_cash_balance, load_credit_card_balance
    pos = load_pos_cash_balance()
    pos_feat = process_pos_cash_features(pos)
    print("POS Features Shape:", pos_feat.shape)
    
    cc = load_credit_card_balance()
    cc_feat = process_credit_card_features(cc)
    print("CC Features Shape:", cc_feat.shape)
