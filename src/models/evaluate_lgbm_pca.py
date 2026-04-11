import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                             recall_score, f1_score, confusion_matrix, classification_report)

def load_data(data_dir=r"d:\IITGN\Project\2 - Invincibles\CodeBase\Data\processed"):
    train = pd.read_csv(os.path.join(data_dir, "train_processed.csv"))
    y = train['TARGET']
    features = [c for c in train.columns if c not in ['TARGET', 'SK_ID_CURR']]
    X = train[features]
    return X, y

def evaluate_lgbm_pca():
    X, y = load_data()
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Identify column types
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    print("Building Preprocessing Pipeline...")
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    # PCA to retain 95% of inherent variance
    pca = PCA(n_components=0.95, random_state=42)
    
    print("Fitting preprocessing & PCA (Expect 1-2 mins on CPU)...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    X_train_pca = pca.fit_transform(X_train_processed)
    X_test_pca = pca.transform(X_test_processed)
    print(f"PCA reduced dimensions from {X_train_processed.shape[1]} to {X_train_pca.shape[1]} components.")
    
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    
    # Matching the optimized config from the main evaluate_lgbm run
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'scale_pos_weight': scale_pos_weight,
        'learning_rate': 0.05,
        'num_leaves': 34,
        'max_depth': 8,
        'min_child_samples': 300,
        'feature_fraction': 0.8,
        'verbose': -1,
        'random_state': 42,
        'device': 'cpu'
    }
    
    print("Training LightGBM model on PCA Components...")
    dtrain = lgb.Dataset(X_train_pca, label=y_train)
    dtest = lgb.Dataset(X_test_pca, label=y_test)
    
    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dtest],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            lgb.log_evaluation(period=50)
        ]
    )
    
    print("\nPredicting on test set...")
    y_pred_proba = model.predict(X_test_pca, num_iteration=model.best_iteration)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\n--- LGBM-PCA EVALUATION METRICS ---")
    print(f"ROC-AUC Score: {auc:.5f}")
    
    print("\nFinding optimal threshold based on F1-Score...")
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_thresh = 0.5
    for t in thresholds:
        preds = (y_pred_proba >= t).astype(int)
        f1 = f1_score(y_test, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            
    print(f"LGBM-PCA Optimal Threshold: {best_thresh:.2f}")
    
    y_pred = (y_pred_proba >= best_thresh).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    
    print(f"Accuracy:  {acc:.5f}")
    print(f"Precision: {prec:.5f}")
    print(f"Recall:    {rec:.5f}")
    print(f"LGBM-PCA F1-Score:  {best_f1:.5f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

if __name__ == "__main__":
    evaluate_lgbm_pca()
