import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                             recall_score, f1_score, confusion_matrix, classification_report)
import os

def load_data(data_dir=r"d:\IITGN\Project\2 - Invincibles\CodeBase\Data\processed"):
    train = pd.read_csv(os.path.join(data_dir, "train_processed.csv"))
    y = train['TARGET']
    features = [c for c in train.columns if c not in ['TARGET', 'SK_ID_CURR']]
    X = train[features].copy()
    
    # LightGBM requires categorical types instead of object strings
    cat_cols = X.select_dtypes(include=['object']).columns
    X[cat_cols] = X[cat_cols].astype('category')
        
    return X, y

def evaluate_model():
    X, y = load_data()
    
    # 80-20 Stratified Split
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    
    # Highly optimized hyperparameter set for Kaggle Home Credit (reduced overfitting, better AUC)
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'scale_pos_weight': scale_pos_weight,
        'learning_rate': 0.02,
        'num_leaves': 34,
        'max_depth': 8,
        'min_child_samples': 300,
        'min_gain_to_split': 0.02,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'random_state': 42,
        'device': 'cpu'
    }
    
    print("Training LightGBM model on train fold...")
    dtrain = lgb.Dataset(X_train, label=y_train)
    dtest = lgb.Dataset(X_test, label=y_test)
    
    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dtest],
        num_boost_round=3000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=150, verbose=True),
            lgb.log_evaluation(period=100)
        ]
    )

    print("\nPredicting on test set...")
    # Get probabilities
    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    
    # ROC AUC
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\n--- EVALUATION METRICS ---")
    print(f"ROC-AUC Score: {auc:.5f}")
    
    # Optimal Threshold tuning based on F1
    print("\nFinding optimal threshold based on F1-Score...")
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_thresh = 0.5
    for t in thresholds:
        preds = (y_pred_proba >= t).astype(int)
        # Handle zero division warnings intentionally by passing zero_division=0
        f1 = f1_score(y_test, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            
    print(f"Optimal Threshold: {best_thresh:.2f}")
    
    # Final Predictions with best threshold
    y_pred = (y_pred_proba >= best_thresh).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    
    print(f"Accuracy:  {acc:.5f}")
    print(f"Precision: {prec:.5f}")
    print(f"Recall:    {rec:.5f}")
    print(f"F1-Score:  {best_f1:.5f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

if __name__ == "__main__":
    evaluate_model()
