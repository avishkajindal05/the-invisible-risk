import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def process_installments_features(inst):
    """
    Section 4.4 Installments Payment Features
    """
    
    inst['PAYMENT_DIFF'] = inst['AMT_INSTALMENT'] - inst['AMT_PAYMENT']
    inst['DAYS_ENTRY_DIFF'] = inst['DAYS_INSTALMENT'] - inst['DAYS_ENTRY_PAYMENT']
    
    inst['LATE_PAYMENT'] = (inst['DAYS_ENTRY_DIFF'] < 0).astype(int)
    inst['SHORT_PAYMENT'] = (inst['PAYMENT_DIFF'] > 0).astype(int)
    
    agg_dict = {
        'SK_ID_PREV': ['count'],
        'PAYMENT_DIFF': ['mean', 'max'],
        'DAYS_ENTRY_DIFF': ['mean'],
        'LATE_PAYMENT': ['mean'],
        'SHORT_PAYMENT': ['mean']
    }
    
    inst_agg = inst.groupby('SK_ID_CURR').agg(agg_dict)
    
    new_cols = []
    for col in inst_agg.columns.values:
        new_cols.append(f"INST_{col[0]}_{col[1].upper()}")
                
    inst_agg.columns = new_cols
    inst_agg = inst_agg.reset_index()
    
    inst_agg.rename(columns={
        'INST_SK_ID_PREV_COUNT': 'INST_COUNT',
        'INST_LATE_PAYMENT_MEAN': 'INST_LATE_PAYMENT_RATE',
        'INST_SHORT_PAYMENT_MEAN': 'INST_SHORT_PAYMENT_RATE'
    }, inplace=True)
    
    # Rolling window for last 365 days (approx filter DAYS_INSTALMENT >= -365)
    inst_365 = inst[inst['DAYS_INSTALMENT'] >= -365].groupby('SK_ID_CURR')['PAYMENT_DIFF'].mean().reset_index()
    inst_365.rename(columns={'PAYMENT_DIFF': 'INST_PAYMENT_DIFF_MEAN_365'}, inplace=True)
    
    inst_agg = inst_agg.merge(inst_365, on='SK_ID_CURR', how='left')
    return inst_agg

if __name__ == "__main__":
    from src.data.loader import load_installments_payments
    inst = load_installments_payments()
    inst_features = process_installments_features(inst)
    print("Installments Features Shape:", inst_features.shape)
