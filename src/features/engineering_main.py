import pandas as pd
import numpy as np
import os

from src.data.loader import (
    load_application_data, load_bureau_data, load_previous_applications,
    load_installments_payments, load_pos_cash_balance, load_credit_card_balance
)
from src.features.engineering_app import process_application_features
from src.features.engineering_bureau import process_bureau_features
from src.features.engineering_prev import process_previous_application_features
from src.features.engineering_installments import process_installments_features
from src.features.engineering_pos_cc import process_pos_cash_features, process_credit_card_features

def build_features(is_train=True):
    """
    Orchestrates the loading, processing, and merging of all features.
    """
    print("Loading Application data...")
    df = load_application_data(is_train=is_train)
    df = process_application_features(df)
    
    print("Loading Bureau data...")
    bureau, bureau_balance = load_bureau_data()
    bureau_feat = process_bureau_features(bureau, bureau_balance)
    df = df.merge(bureau_feat, on='SK_ID_CURR', how='left')
    
    print("Loading Previous Applications data...")
    prev = load_previous_applications()
    prev_feat = process_previous_application_features(prev)
    df = df.merge(prev_feat, on='SK_ID_CURR', how='left')
    
    print("Loading Installments data...")
    inst = load_installments_payments()
    inst_feat = process_installments_features(inst)
    df = df.merge(inst_feat, on='SK_ID_CURR', how='left')
    
    print("Loading POS and CC data...")
    pos = load_pos_cash_balance()
    pos_feat = process_pos_cash_features(pos)
    df = df.merge(pos_feat, on='SK_ID_CURR', how='left')
    
    cc = load_credit_card_balance()
    cc_feat = process_credit_card_features(cc)
    df = df.merge(cc_feat, on='SK_ID_CURR', how='left')
    
    print("Shape after merges:", df.shape)
    
    # Store the merged dataframe to disk to avoid re-computation
    output_dir = r"d:\IITGN\Project\2 - Invincibles\CodeBase\Data\processed"
    os.makedirs(output_dir, exist_ok=True)
    out_name = "train_processed.csv" if is_train else "test_processed.csv"
    df.to_csv(os.path.join(output_dir, out_name), index=False)
    
    return df

if __name__ == "__main__":
    df_train = build_features(is_train=True)
    df_test = build_features(is_train=False)
