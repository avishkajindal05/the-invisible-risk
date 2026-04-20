import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
import numpy as np
import os
import re
import warnings
warnings.filterwarnings('ignore')

def sanitize_columns(cols):
    """Replace special characters in column names that LightGBM rejects."""
    return [re.sub(r'[^A-Za-z0-9_]', '_', c) for c in cols]

def evaluate(clf, X_test, y_test):
    proba = clf.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, proba)
    best_f1, best_thresh = 0, 0.5
    for thresh in np.arange(0.30, 0.70, 0.02):
        f1 = f1_score(y_test, (proba > thresh).astype(int))
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh
    return roc, best_f1, best_thresh

def make_lgbm():
    return lgb.LGBMClassifier(
        n_estimators=400,
        num_leaves=96,
        learning_rate=0.05,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )

def train():
    print("=" * 60)
    print("  CREDIT DEFAULT — DUAL MODEL TRAINING  ")
    print("=" * 60)

    # ---- Load ----
    print("\n[1/7] Loading processed data...")
    df = pd.read_csv('Data/processed/train_processed.csv')
    print(f"      Rows: {len(df):,}  |  Columns: {len(df.columns)}")

    y = df['TARGET'].values

    # ---- Encode ----
    object_cols = df.select_dtypes(['object']).columns.tolist()
    print(f"\n[2/7] One-hot encoding {len(object_cols)} categorical features...")
    df_enc = pd.get_dummies(df.drop(columns=['TARGET', 'SK_ID_CURR']), columns=object_cols)

    # Sanitize column names (LightGBM rejects special chars)
    df_enc.columns = sanitize_columns(df_enc.columns.tolist())
    print(f"      Total encoded features: {df_enc.shape[1]}")

    # ---- Impute ----
    print("\n[3/7] Imputing missing values with column median...")
    medians = df_enc.median()
    df_enc = df_enc.fillna(medians).fillna(0)  # second fillna covers cols where median is NaN

    X_full = df_enc.values
    feature_names_full = df_enc.columns.tolist()

    # ---- MODEL A : FULL ----
    print("\n" + "=" * 60)
    print("  MODEL A — Full Features (Existing Applicants)")
    print("=" * 60)

    print("[4/7] Applying SMOTE...")
    sm = SMOTE(random_state=42)
    X_sm, y_sm = sm.fit_resample(X_full, y)
    print(f"      After SMOTE: {len(X_sm):,} samples")

    X_tr, X_te, y_tr, y_te = train_test_split(X_sm, y_sm, test_size=0.2, random_state=42)
    clf_full = make_lgbm()
    clf_full.fit(X_tr, y_tr)

    roc_f, f1_f, thresh_f = evaluate(clf_full, X_te, y_te)
    print(f"      [OK] ROC AUC:  {roc_f:.4f}")
    print(f"      [OK] F1 Score: {f1_f:.4f}  (thresh={thresh_f:.2f})")

    # ---- MODEL B : LITE ----
    print("\n" + "=" * 60)
    print("  MODEL B — Lite Features (New Applicants)")
    print("=" * 60)

    lite_raw = [
        'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
        'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
        'AMT_GOODS_PRICE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
        'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'DAYS_BIRTH', 'DAYS_EMPLOYED'
    ]
    lite_sanitized = sanitize_columns(lite_raw)

    # Find all encoded columns that belong to a lite feature
    lite_cols = [
        c for c in feature_names_full
        if any(c == s or c.startswith(s + '_') for s in lite_sanitized)
    ]
    print(f"      Lite encoded feature count: {len(lite_cols)}")

    lite_idx = [feature_names_full.index(c) for c in lite_cols]
    X_lite = X_full[:, lite_idx]

    print("[5/7] Applying SMOTE (lite)...")
    sm_lite = SMOTE(random_state=42)
    X_sm_l, y_sm_l = sm_lite.fit_resample(X_lite, y)
    print(f"      After SMOTE: {len(X_sm_l):,} samples")

    X_tr_l, X_te_l, y_tr_l, y_te_l = train_test_split(X_sm_l, y_sm_l, test_size=0.2, random_state=42)
    clf_lite = make_lgbm()
    clf_lite.fit(X_tr_l, y_tr_l)

    roc_l, f1_l, thresh_l = evaluate(clf_lite, X_te_l, y_te_l)
    print(f"      [OK] ROC AUC:  {roc_l:.4f}")
    print(f"      [OK] F1 Score: {f1_l:.4f}  (thresh={thresh_l:.2f})")

    # ---- SAVE ----
    print("\n[6/7] Saving models...")
    os.makedirs('models', exist_ok=True)

    joblib.dump({
        'model': clf_full,
        'features': feature_names_full,
        'best_thresh': thresh_f
    }, 'models/lgb_model_full.pkl')

    joblib.dump({
        'model': clf_lite,
        'features': lite_cols,
        'best_thresh': thresh_l
    }, 'models/lgb_model_lite.pkl')

    print("[7/7] Both models saved OK.")

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print(f"  Model A (Full):  F1={f1_f:.4f}  AUC={roc_f:.4f}")
    print(f"  Model B (Lite):  F1={f1_l:.4f}  AUC={roc_l:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    train()
